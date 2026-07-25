#!/usr/bin/env python3
"""Generate a CPU harness for any nvjpeg format_to_ycbcr variant.
Usage: gen_format_to_ycbcr.py <infmt 3-6> <subsamp 0-6> > host_cpu.cpp
 infmt: 3=RGB(planar) 4=BGR(planar) 5=RGBI(interleaved) 6=BGRI(interleaved)
 subsamp: 0=444 1=422 2=420 3=440 4=411 5=410 6=GRAY
RGB->Y/Cb/Cr (JFIF). Y full-res; Cb/Cr = avg of clamped chroma over the subsampling
block (block_w x block_h). const-bank: in@0x210(ch0/1/2 @0x210/18/20, pitch @0x230/38/40),
inSize@0x250, out@0x258(ch0/1/2, pitch @0x278/80/88), roi@0x298=0.
"""
import sys
# subsamp -> (block_w, block_h)  (chroma block: cols, rows of pixels per chroma sample)
BLK = {0:(1,1), 1:(2,1), 2:(2,2), 3:(1,2), 4:(4,1), 5:(4,2)}

def main():
    infmt, sub = int(sys.argv[1]), int(sys.argv[2])
    gray = (sub == 6)
    interleaved = infmt in (5, 6)
    # channel order in source: which of (R,G,B) is at slot 0,1,2
    # planar RGB(3)/RGBI(5): slot0=R,1=G,2=B ; BGR(4)/BGRI(6): slot0=B,1=G,2=R
    rgb_first = infmt in (3, 5)
    W, H = 16, 8
    mg = (f"_ZN6nvjpeg22format_to_ycbcr_kernelIL19nvjpegInputFormat_t{infmt}E"
          f"L25nvjpegChromaSubsampling_t{sub}ENS_15ColorModelToYUV12"
          f"LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_")
    if gray:
        bw = bh = 1
    else:
        bw, bh = BLK[sub]
    CW, CH = W // bw, H // bh
    Ndiv = bw * bh

    # source pixel accessors return R,G,B given (r,c)
    if interleaved:
        in_decl = f"uint8_t IN[{H}*{W}*3]; for(int r=0;r<{H};++r)for(int c=0;c<{W};++c){{ IN[(r*{W}+c)*3+0]=(uint8_t)(r*23+c*11+7); IN[(r*{W}+c)*3+1]=(uint8_t)(r*31+c*17+40); IN[(r*{W}+c)*3+2]=(uint8_t)(r*13+c*29+200); }}"
        sl0, sl1, sl2 = ("R", "G", "B") if rgb_first else ("B", "G", "R")
        getr = f"float {sl0}=(float)IN[(r*{W}+c)*3+0], {sl1}=(float)IN[(r*{W}+c)*3+1], {sl2}=(float)IN[(r*{W}+c)*3+2];"
        in_args = ("        ArgAt{0x210},(uint8_t*)IN,\n"
                   f"        ArgAt{{0x230}},(uint64_t){W*3},\n")
    else:
        # planar: 3 buffers; slot0=ch0@0x210 etc.
        s0, s1, s2 = ("R", "G", "B") if rgb_first else ("B", "G", "R")
        in_decl = (f"uint8_t P0[{H}*{W}],P1[{H}*{W}],P2[{H}*{W}];\n"
                   f"    for(int r=0;r<{H};++r)for(int c=0;c<{W};++c){{ P0[r*{W}+c]=(uint8_t)(r*23+c*11+7); P1[r*{W}+c]=(uint8_t)(r*31+c*17+40); P2[r*{W}+c]=(uint8_t)(r*13+c*29+200); }}")
        getr = f"float {s0}=(float)P0[r*{W}+c], {s1}=(float)P1[r*{W}+c], {s2}=(float)P2[r*{W}+c];"
        in_args = ("        ArgAt{0x210},(uint8_t*)P0, ArgAt{0x218},(uint8_t*)P1, ArgAt{0x220},(uint8_t*)P2,\n"
                   f"        ArgAt{{0x230}},(uint64_t){W}, ArgAt{{0x238}},(uint64_t){W}, ArgAt{{0x240}},(uint64_t){W},\n")

    if gray:
        out_decl = f"uint8_t oY[{H}*{W}],rY[{H}*{W}]; for(int i=0;i<{H}*{W};++i)oY[i]=0xCC;"
        ref = f"""    for(int r=0;r<{H};++r)for(int c=0;c<{W};++c){{ {getr} rY[r*{W}+c]=ct(Yf(R,G,B)); }}"""
        out_args = ("        ArgAt{0x258},(uint8_t*)oY,\n"
                    f"        ArgAt{{0x278}},(uint64_t){W},\n")
        cmp = f"int bad=cmp_u8_tol(oY,rY,{H}*{W},1,\"Y\");"
        ntot = f"{H}*{W}"
        launch_block = f"Dim3{{32,8}}, Dim3{{(({W}+31)/32),(({H}+7)/8}}"  # placeholder, replaced below
    else:
        out_decl = (f"uint8_t oY[{H}*{W}],oCb[{CH}*{CW}],oCr[{CH}*{CW}],rY[{H}*{W}],rCb[{CH}*{CW}],rCr[{CH}*{CW}];\n"
                    f"    for(int i=0;i<{H}*{W};++i)oY[i]=0xCC; for(int i=0;i<{CH}*{CW};++i){{oCb[i]=0xCC;oCr[i]=0xCC;}}")
        ref = f"""    for(int r=0;r<{H};++r)for(int c=0;c<{W};++c){{ {getr} rY[r*{W}+c]=ct(Yf(R,G,B)); }}
    for(int cy=0;cy<{CH};++cy)for(int cx=0;cx<{CW};++cx){{
        float sb=0,sr=0;
        for(int dy=0;dy<{bh};++dy)for(int dx=0;dx<{bw};++dx){{ int r={bh}*cy+dy,c={bw}*cx+dx; {getr} sb+=(float)ct(Cbf(R,G,B)); sr+=(float)ct(Crf(R,G,B)); }}
        rCb[cy*{CW}+cx]=ct(sb*{1.0/Ndiv}f); rCr[cy*{CW}+cx]=ct(sr*{1.0/Ndiv}f);
    }}"""
        out_args = ("        ArgAt{0x258},(uint8_t*)oY, ArgAt{0x260},(uint8_t*)oCb, ArgAt{0x268},(uint8_t*)oCr,\n"
                    f"        ArgAt{{0x278}},(uint64_t){W}, ArgAt{{0x280}},(uint64_t){CW}, ArgAt{{0x288}},(uint64_t){CW},\n")
        cmp = (f"int bad=cmp_u8_tol(oY,rY,{H}*{W},1,\"Y\")+cmp_u8_tol(oCb,rCb,{CH}*{CW},1,\"Cb\")+cmp_u8_tol(oCr,rCr,{CH}*{CW},1,\"Cr\");")
        ntot = f"{H}*{W}+2*{CH}*{CW}"

    print(f'''// AUTO-GENERATED (gen_format_to_ycbcr.py {infmt} {sub}) infmt={infmt} subsamp={sub}
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"
#include "nvlib_compare.h"
extern "C" void {mg}();
static uint8_t ct(float v){{ v=v>=0?v:0; v=v<255?v:255; return (uint8_t)(uint32_t)truncf(v); }}
static float Yf (float R,float G,float B){{ return fmaf(B,0.11400000005960464478f, fmaf(R,0.29899999499320983887f, G*0.58700001239776611328f)); }}
static float Cbf(float R,float G,float B){{ return fmaf(B,0.5f, fmaf(R,-0.16873589158058166504f, -(G*0.33126410841941833496f))) + 128.0f; }}
static float Crf(float R,float G,float B){{ return fmaf(B,-0.081312410533428192139f, fmaf(R,0.5f, -(G*0.41868758201599121094f))) + 128.0f; }}
int main(){{
    const int W={W},H={H};
    {in_decl}
    {out_decl}
{ref}
    launchKernel2D({mg}, Dim3{{1,1}}, Dim3{{{CW},{CH}}},
{in_args}        ArgAt{{0x250}},(int)W, ArgAt{{0x254}},(int)H,
{out_args}        ArgAt{{0x298}},(int)0, ArgAt{{0x29C}},(int)0);
    {cmp}
    return report(bad,{ntot});
}}''')

if __name__ == "__main__":
    main()
