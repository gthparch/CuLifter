.visible .entry check_ldtm(.param .u64 output)
.reqntid 32, 1, 1
{
    .shared .align 4 .b32 slot;
    .reg .b32 %slot_addr, %base, %lane, %v<8>;
    .reg .b64 %out, %offset;
    mov.u32 %slot_addr, slot;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%slot_addr], 32;
    bar.sync 0;
    ld.shared.u32 %base, [%slot_addr];

    mov.u32 %lane, %laneid;
    mad.lo.u32 %v0, %lane, 16, 4096;
    add.u32 %v1, %v0, 1;
    add.u32 %v2, %v0, 2;
    add.u32 %v3, %v0, 3;
    add.u32 %v4, %v0, 4;
    add.u32 %v5, %v0, 5;
    add.u32 %v6, %v0, 6;
    add.u32 %v7, %v0, 7;
    tcgen05.st.sync.aligned.32x32b.x8.b32 [%base], {%v0, %v1, %v2, %v3, %v4, %v5, %v6, %v7};
    tcgen05.wait::st.sync.aligned;

    ld.param.u64 %out, [output];
    mul.wide.u32 %offset, %lane, 24;
    add.u64 %out, %out, %offset;
    {
        .param .b32 addr;
        .param .b64 dst;
        st.param.b32 [addr], %base;
        st.param.b64 [dst], %out;
        call.uni read_tmem, (addr, dst);
    }

    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %base, 32;
    tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
    ret;
}
