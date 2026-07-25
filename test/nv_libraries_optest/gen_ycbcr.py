#!/usr/bin/env python3
"""Generate a CPU exec-correctness harness for any nvjpeg ycbcr_to_format variant.

Usage: gen_ycbcr.py <subsamp 0-7> <fmt 3-6> > host_cpu.cpp
  subsamp: 0=444 1=422 2=420 3=440 4=411 5=410 6=GRAY 7=410V
  fmt:     3=RGB(planar) 4=BGR(planar) 5=RGBI(interleaved) 6=BGRI(interleaved)
Planar YCbCr in (ch0=Y@0x210, ch1=Cb@0x218, ch2=Cr@0x220, pitch@0x230/38/40),
inSize@0x250, out@0x258(+pitch@0x278/80/88), roi@0x298=0. FFMA conversion (fmaf),
±1 tolerance on the FP-derived uint8 outputs.
"""
import sys

# (chroma row-shift, col-shift) per subsampling
SUB = {0:(0,0), 1:(0,1), 2:(1,1), 3:(1,0), 4:(0,2), 5:(1,2), 6:None, 7:(2,1)}

def mangled(s, f):
    return (f"_ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t{s}E"
            f"Lb0EL20nvjpegOutputFormat_t{f}ENS_26ConvertToFormatDispatchROI12"
            f"LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_")

def main():
    s, f = int(sys.argv[1]), int(sys.argv[2])
    mg = mangled(s, f)
    gray = (s == 6)
    interleaved = (f in (5, 6))
    # output channel order: RGB(3)/RGBI(5) -> R,G,B ; BGR(4)/BGRI(6) -> B,G,R
    rgb_order = f in (3, 5)
    W, H = 16, 8
    if gray:
        crow = ccol = 0; CW, CH = W, H
    else:
        rsh, csh = SUB[s]; CW, CH = W >> csh, H >> rsh

    # reference assignment per output channel index 0,1,2
    if rgb_order:  ch = ["R", "G", "B"]
    else:          ch = ["B", "G", "R"]

    chroma_decl = "" if gray else f"""
    uint8_t Cb[{CH}*{CW}], Cr[{CH}*{CW}];
    for(int r=0;r<{CH};++r)for(int c=0;c<{CW};++c){{ Cb[r*{CW}+c]=(uint8_t)(r*31+c*17+40); Cr[r*{CW}+c]=(uint8_t)(r*13+c*29+200); }}"""
    if gray:
        conv = "float R=Yf,G=Yf,B=Yf;"
    else:
        ci = f"((r>>{SUB[s][0]})*{CW} + (c>>{SUB[s][1]}))"
        conv = (f"float Cbm=(float)Cb[{ci}]-128, Crm=(float)Cr[{ci}]-128;\n"
                f"        float B=fmaf(Cbm,1.7719999551773071289f,Yf);\n"
                f"        float G=fmaf(Crm,-0.71413630247116088867f,fmaf(Cbm,-0.34413629770278930664f,Yf));\n"
                f"        float R=fmaf(Crm,1.4019999504089355469f,Yf);")
    # chroma plane marshal (skip for gray but harmless to include real buffers)
    chroma_args = "" if gray else "        ArgAt{0x218},(uint8_t*)Cb, ArgAt{0x220},(uint8_t*)Cr,\n        ArgAt{0x238},(uint64_t)%d, ArgAt{0x240},(uint64_t)%d,\n" % (CW, CW)
    if gray:
        # gray kernel still has ch1/ch2 params; point them at Y to be safe
        chroma_args = "        ArgAt{0x218},(uint8_t*)Y, ArgAt{0x220},(uint8_t*)Y,\n        ArgAt{0x238},(uint64_t)%d, ArgAt{0x240},(uint64_t)%d,\n" % (W, W)

    if interleaved:
        out_decl = f"uint8_t O[{H}*{W}*3]; for(int i=0;i<{H}*{W}*3;++i)O[i]=0xCC;"
        out_args = ("        ArgAt{0x258},(uint8_t*)O,\n"
                    f"        ArgAt{{0x278}},(uint64_t){W*3},\n")
        store = (f"        O[(r*{W}+c)*3+0]=ct({ch[0]}); O[(r*{W}+c)*3+1]=ct({ch[1]}); O[(r*{W}+c)*3+2]=ct({ch[2]});")
        cmp = (f"int bad=0; uint8_t g0[{H}*{W}],g1[{H}*{W}],g2[{H}*{W}],r0[{H}*{W}],r1[{H}*{W}],r2[{H}*{W}];\n"
               f"    for(int i=0;i<{H}*{W};++i){{g0[i]=O[i*3];g1[i]=O[i*3+1];g2[i]=O[i*3+2];r0[i]=REF[i*3];r1[i]=REF[i*3+1];r2[i]=REF[i*3+2];}}\n"
               f"    bad=cmp_u8_tol(g0,r0,{H}*{W},1,\"c0\")+cmp_u8_tol(g1,r1,{H}*{W},1,\"c1\")+cmp_u8_tol(g2,r2,{H}*{W},1,\"c2\");")
        ref_decl = f"uint8_t REF[{H}*{W}*3];"
        ref_store = f"        REF[(r*{W}+c)*3+0]=ct({ch[0]}); REF[(r*{W}+c)*3+1]=ct({ch[1]}); REF[(r*{W}+c)*3+2]=ct({ch[2]});"
        ntot = f"3*{H}*{W}"
    else:
        out_decl = f"uint8_t O0[{H}*{W}],O1[{H}*{W}],O2[{H}*{W}]; for(int i=0;i<{H}*{W};++i)O0[i]=O1[i]=O2[i]=0xCC;"
        out_args = ("        ArgAt{0x258},(uint8_t*)O0, ArgAt{0x260},(uint8_t*)O1, ArgAt{0x268},(uint8_t*)O2,\n"
                    f"        ArgAt{{0x278}},(uint64_t){W}, ArgAt{{0x280}},(uint64_t){W}, ArgAt{{0x288}},(uint64_t){W},\n")
        store = f"        O0[r*{W}+c]=ct({ch[0]}); O1[r*{W}+c]=ct({ch[1]}); O2[r*{W}+c]=ct({ch[2]});"
        cmp = (f"int bad=cmp_u8_tol(O0,R0,{H}*{W},1,\"c0\")+cmp_u8_tol(O1,R1,{H}*{W},1,\"c1\")+cmp_u8_tol(O2,R2,{H}*{W},1,\"c2\");")
        ref_decl = f"uint8_t R0[{H}*{W}],R1[{H}*{W}],R2[{H}*{W}];"
        ref_store = f"        R0[r*{W}+c]=ct({ch[0]}); R1[r*{W}+c]=ct({ch[1]}); R2[r*{W}+c]=ct({ch[2]});"
        ntot = f"3*{H}*{W}"

    print(f'''// AUTO-GENERATED (gen_ycbcr.py {s} {f}) — ycbcr_to_format subsamp={s} fmt={f}
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"
#include "nvlib_compare.h"
extern "C" void {mg}();
static uint8_t ct(float v){{ v=v>=0?v:0; v=v<255?v:255; return (uint8_t)(uint32_t)truncf(v); }}
int main(){{
    const int W={W},H={H};
    uint8_t Y[H*W]; for(int r=0;r<H;++r)for(int c=0;c<W;++c) Y[r*W+c]=(uint8_t)(r*23+c*11+7);{chroma_decl}
    {ref_decl}
    for(int r=0;r<H;++r)for(int c=0;c<W;++c){{
        float Yf=Y[r*W+c];
        {conv}
{ref_store}
    }}
    {out_decl}
    launchKernel2D({mg}, Dim3{{(W+31)/32,(H+7)/8}}, Dim3{{32,8}},
        ArgAt{{0x210}},(uint8_t*)Y,
{chroma_args}        ArgAt{{0x230}},(uint64_t)W,
        ArgAt{{0x250}},(int)W, ArgAt{{0x254}},(int)H,
{out_args}        ArgAt{{0x298}},(int)0, ArgAt{{0x29C}},(int)0);
    {cmp}
    return report(bad,{ntot});
}}''')

if __name__ == "__main__":
    main()
