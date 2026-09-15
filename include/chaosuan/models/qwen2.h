#ifndef CHAOSUAN_MODELS_QWEN2_H
#define CHAOSUAN_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct ChaosuanQwen2Meta {
        chaosuanDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct ChaosuanQwen2Weights {
        chaosuanTensor_t in_embed;
        chaosuanTensor_t out_embed;
        chaosuanTensor_t out_norm_w;   // a.k.a. model.norm.weight
        chaosuanTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        chaosuanTensor_t *attn_q_w;
        chaosuanTensor_t *attn_q_b;
        chaosuanTensor_t *attn_k_w;
        chaosuanTensor_t *attn_k_b;
        chaosuanTensor_t *attn_v_w;
        chaosuanTensor_t *attn_v_b;
        chaosuanTensor_t *attn_o_w;
        chaosuanTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        chaosuanTensor_t *mlp_gate_w;
        chaosuanTensor_t *mlp_up_w;
        chaosuanTensor_t *mlp_down_w;
    };

    struct ChaosuanQwen2Model;

    __export struct ChaosuanQwen2Model *chaosuanQwen2ModelCreate(const ChaosuanQwen2Meta *meta, chaosuanDeviceType_t device, int *device_ids, int ndevice);

    __export void chaosuanQwen2ModelDestroy(struct ChaosuanQwen2Model * model);

    __export struct ChaosuanQwen2Weights *chaosuanQwen2ModelWeights(struct ChaosuanQwen2Model * model);

    __export int64_t chaosuanQwen2ModelInfer(struct ChaosuanQwen2Model * model, int64_t * token_ids, size_t ntoken);
}
#endif // CHAOSUAN_MODELS_QWEN2_H
