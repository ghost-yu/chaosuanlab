#ifndef CHAOSUAN_OPS_H
#define CHAOSUAN_OPS_H

#include "tensor.h"

__C {
    __export void chaosuanAdd(chaosuanTensor_t c, chaosuanTensor_t a, chaosuanTensor_t b);
    __export void chaosuanArgmax(chaosuanTensor_t max_idx, chaosuanTensor_t max_val, chaosuanTensor_t vals);
    __export void chaosuanEmbedding(chaosuanTensor_t out, chaosuanTensor_t index, chaosuanTensor_t weight);
    __export void chaosuanLinear(chaosuanTensor_t out, chaosuanTensor_t in, chaosuanTensor_t weight, chaosuanTensor_t bias);
    __export void chaosuanRearrange(chaosuanTensor_t out, chaosuanTensor_t in);
    __export void chaosuanRmsNorm(chaosuanTensor_t out, chaosuanTensor_t in, chaosuanTensor_t weight, float eps);
    __export void chaosuanROPE(chaosuanTensor_t out, chaosuanTensor_t in, chaosuanTensor_t pos_ids, float theta);
    __export void chaosuanSelfAttention(chaosuanTensor_t attn_val, chaosuanTensor_t q, chaosuanTensor_t k, chaosuanTensor_t v, float scale);
    __export void chaosuanSwiGLU(chaosuanTensor_t out, chaosuanTensor_t gate, chaosuanTensor_t up);
}

#endif
