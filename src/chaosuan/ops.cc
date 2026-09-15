#include "chaosuan/ops.h"

#include "chaosuan_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

__C {
    void chaosuanAdd(chaosuanTensor_t c, chaosuanTensor_t a, chaosuanTensor_t b) {
        chaosuan::ops::add(c->tensor, a->tensor, b->tensor);
    }
    void chaosuanArgmax(chaosuanTensor_t max_idx, chaosuanTensor_t max_val, chaosuanTensor_t vals) {
        chaosuan::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor);
    }
    void chaosuanEmbedding(chaosuanTensor_t out, chaosuanTensor_t index, chaosuanTensor_t weight) {
        chaosuan::ops::embedding(out->tensor, index->tensor, weight->tensor);
    }
    void chaosuanLinear(chaosuanTensor_t out, chaosuanTensor_t in, chaosuanTensor_t weight, chaosuanTensor_t bias) {
        chaosuan::ops::linear(out->tensor, in->tensor, weight->tensor, bias->tensor);
    }
    void chaosuanRearrange(chaosuanTensor_t out, chaosuanTensor_t in) {
        chaosuan::ops::rearrange(out->tensor, in->tensor);
    }
    void chaosuanRmsNorm(chaosuanTensor_t out, chaosuanTensor_t in, chaosuanTensor_t weight, float eps) {
        chaosuan::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps);
    }
    void chaosuanROPE(chaosuanTensor_t out, chaosuanTensor_t in, chaosuanTensor_t pos_ids, float theta) {
        chaosuan::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta);
    }
    void chaosuanSelfAttention(chaosuanTensor_t attn_val, chaosuanTensor_t q, chaosuanTensor_t k, chaosuanTensor_t v, float scale) {
        chaosuan::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
    }
    void chaosuanSwiGLU(chaosuanTensor_t out, chaosuanTensor_t gate, chaosuanTensor_t up) {
        chaosuan::ops::swiglu(out->tensor, gate->tensor, up->tensor);
    }
}
