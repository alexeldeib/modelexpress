"""Patch NIXL ucx_utils.cpp to fix VRAM detection in subprocess environments.

1. Set explicit CUDA memory type in ucp_mem_map (bypass auto-detection)
2. Remove the ucp_mem_query HOST check (unreliable in subprocess environments)
"""
import re
import sys

path = sys.argv[1]
with open(path) as f:
    code = f.read()

# 1. Replace the mem_params initialization to add explicit memory type
old_params = """    ucp_mem_map_params_t mem_params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_FLAGS | UCP_MEM_MAP_PARAM_FIELD_LENGTH |
            UCP_MEM_MAP_PARAM_FIELD_ADDRESS,
        .address = mem.base,
        .length = mem.size,
    };"""

new_params = """    ucp_mem_map_params_t mem_params;
    memset(&mem_params, 0, sizeof(mem_params));
    mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_ADDRESS;
    mem_params.address = mem.base;
    mem_params.length = mem.size;
    if (nixl_mem_type == nixl_mem_t::VRAM_SEG) {
        mem_params.field_mask |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        mem_params.memory_type = UCS_MEMORY_TYPE_CUDA;
    }"""

assert old_params in code, "Could not find mem_params initialization"
code = code.replace(old_params, new_params)

# 2. Remove the VRAM verification block (ucp_mem_query + HOST check)
old_check = """    if (nixl_mem_type == nixl_mem_t::VRAM_SEG) {
        ucp_mem_attr_t attr;
        attr.field_mask = UCP_MEM_ATTR_FIELD_MEM_TYPE;
        status = ucp_mem_query(mem.memh, &attr);
        if (status != UCS_OK) {
            NIXL_ERROR << "Failed to ucp_mem_query: " << ucs_status_string(status);
            ucp_mem_unmap(ctx, mem.memh);
            return -1;
        }

        if (attr.mem_type == UCS_MEMORY_TYPE_HOST) {
            NIXL_ERROR << "VRAM memory is detected as host by UCX. "
                          "UCX is likely not configured with CUDA support. "
                          "VRAM registration cannot proceed.";
            ucp_mem_unmap(ctx, mem.memh);
            return -1;
        }
    }"""

assert old_check in code, "Could not find VRAM verification block"
code = code.replace(old_check, "    // VRAM check removed: explicit memory type set above")

with open(path, 'w') as f:
    f.write(code)

print("NIXL VRAM fix applied successfully")
