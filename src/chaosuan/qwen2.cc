// ============================================================================
// qwen2.cc —— Qwen2 大语言模型的 C++ 后端实现（作业 3）
// ----------------------------------------------------------------------------
// 【本文件做什么？】
// 这是整个作业 3 的核心：用我们作业 1、2 写的"张量 + 算子"，
// 拼出一个真正的、能跑 DeepSeek-R1-Distill-Qwen-1.5B 的 Transformer！
//
// 【模型前向完整流程】（对应 test_infer.py 里的 PyTorch 版本）
//   输入 token_ids（一串词编号，比如 [151644, 8948, ...]）
//     ↓
//   1. embedding：查词向量表，得到 hidden = [ntoken, hs]
//     ↓
//   2. 循环 28 层（num_hidden_layers）：
//      a. rms_norm(hidden, attn_norm_w)          → 注意力前的归一化
//      b. linear → q / k / v（三套权重和偏置）
//      c. reshape q/k/v 成 [ntoken, nhead, dh]
//      d. rope(q), rope(k)                        → 旋转位置编码
//      e. 把新的 k/v 写入 KV-Cache（关键！避免重复算历史）
//      f. self_attention(q, k_cache, v_cache)     → 注意力输出
//      g. linear(attn, o_w) + 残差连接            → hidden += o
//      h. rms_norm + SwiGLU(gate/up) + down + 残差 → MLP
//     ↓
//   3. 最后一层 rms_norm(hidden, out_norm_w)
//     ↓
//   4. linear(hidden, out_embed)                  → logits [ntoken, vocab]
//     ↓
//   5. argmax(最后一行 logits)                     → 下一个 token
//
// 【KV-Cache 是什么？为什么必须有？】
// 生成第 100 个 token 时，如果每次把全部 100 个 token 重新算一遍前向，
// 耗时是 O(n²)。KV-Cache 的做法：把历史每个 token 的 k/v 向量**存下来**，
// 新 token 只需要算自己那一步的点积，历史部分直接复用缓存 → 耗时 O(n)。
// 我们的 self_attention 算子天然支持"total_len > seqlen"（见作业 2.6），
// 只要把缓存行数作为 total_len 传进去即可。
//
// 【tie_word_embeddings（权重共享）】
// 这个模型的 lm_head 与 embedding 共享同一张权重表（config 里
// tie_word_embeddings=true），所以 logits = hidden @ embed_tokens^T。
// 这就是为什么 Weights 结构里 in_embed 和 out_embed 指向同一个张量。
// ============================================================================

#include "chaosuan/models/qwen2.h"  // C API 原型（本文件实现它）。

#include "chaosuan_tensor.hpp"      // C API 层的 tensor 包装（ChaosuanTensor）。

// 作业 2 实现的全部算子：
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include "../tensor/tensor.hpp"    // 张量类（create/view/slice/load）。
#include "../utils.hpp"            // 类型转换、异常宏。

#include <algorithm>  // std::sort / std::unique（释放权重前去重）/ std::partial_sort。
#include <cmath>    // std::sqrt（注意力缩放 1/sqrt(dh)）。
#include <cstring>  // std::memcpy（写 KV-Cache 用）。
#include <iostream> // std::cerr（临时调试输出）。
#include <utility>  // std::pair（调试用）。
#include <vector>   // std::vector（动态数组，存每层的权重和缓存）。

namespace chaosuan {
namespace models {

// 私有命名空间：只在本文件内可见的辅助函数。
namespace {

// 创建一个全 1 的 I64 张量数据缓冲区并返回（给 pos_ids 用）。
// 我们直接操作原始内存，不需要经过 Tensor 的 load。
std::vector<int64_t> make_pos_ids(size_t start, size_t n) {
    std::vector<int64_t> pos(n);
    for (size_t i = 0; i < n; i++) pos[i] = static_cast<int64_t>(start + i);
    return pos;
}

} // namespace

// ============================================================================
// Qwen2Model —— 模型内部实现类
// ----------------------------------------------------------------------------
// 为什么不直接写在 C API 函数里？因为推理要跨多次调用保存状态：
//   - 所有层的权重；
//   - KV-Cache 当前长度（cache_len）；
//   - 每个张量的设备信息。
// 用一个类把这些状态包起来，C API 里只保存一个指向它的指针。
// ============================================================================
struct Qwen2Model {
    // ---- 超参数（来自 config.json，经 ChaosuanQwen2Meta 传入）----
    chaosuanDataType_t dtype;  // 权重数据类型（本项目用 bf16，与 HF 加载一致）
    size_t nlayer;             // Transformer 层数（28）
    size_t hs;                 // hidden_size（1536）
    size_t nh;                 // 注意力头数（12）
    size_t nkvh;               // KV 头数（GQA，2）
    size_t dh;                 // 每个头的维度（128）
    size_t di;                 // MLP 中间维度（8960）
    size_t maxseq;             // 最大序列长度（KV-Cache 容量）
    size_t voc;                // 词表大小（151936）
    float epsilon;             // RMSNorm 的 eps
    float theta;               // RoPE 的频率基
    int64_t end_token;         // 结束符（生成到它就停）
    chaosuanDeviceType_t device;
    int device_id;

    // ---- 权重填充入口 ----
    // Python 侧通过 chaosuanQwen2ModelWeights() 拿到这个结构，
    // 把每个权重张量（tensorCreate + load 好的）填进去；
    // 第一次 infer 时我们才真正"收编"这些张量（惰性初始化）。
    ChaosuanQwen2Weights weights;  // 初始全为 null
    bool weights_ready = false;

    // ---- 所有权管理 ----
    // C API 传入的权重张量（chaosuanTensor_t 原始指针）由本模型统一释放。
    // 注意 in_embed 和 out_embed 可能是同一个指针（权重共享），
    // 所以释放前要去重，避免重复 delete。
    std::vector<chaosuanTensor_t> owned_weights;

    // ---- 权重张量（tensor_t = shared_ptr<Tensor>，作业 1 的成果）----
    tensor_t in_embed;   // 词向量表 [voc, hs]
    tensor_t out_embed;  // lm_head（共享 in_embed）[voc, hs]
    tensor_t out_norm_w; // 最后一层归一化权重 [hs]
    // 每层的权重（下标 = 层号）：
    std::vector<tensor_t> attn_norm_w; // 注意力前归一化 [hs]
    std::vector<tensor_t> attn_q_w;    // q 投影 [nh*dh, hs]
    std::vector<tensor_t> attn_q_b;    // q 偏置 [nh*dh]
    std::vector<tensor_t> attn_k_w;    // k 投影 [nkvh*dh, hs]
    std::vector<tensor_t> attn_k_b;    // k 偏置 [nkvh*dh]
    std::vector<tensor_t> attn_v_w;    // v 投影 [nkvh*dh, hs]
    std::vector<tensor_t> attn_v_b;    // v 偏置 [nkvh*dh]
    std::vector<tensor_t> attn_o_w;    // o 投影 [hs, nh*dh]（Qwen2 无 bias）
    std::vector<tensor_t> mlp_norm_w;  // MLP 前归一化 [hs]
    std::vector<tensor_t> mlp_gate_w;  // SwiGLU 门控 [di, hs]
    std::vector<tensor_t> mlp_up_w;    // SwiGLU 上升 [di, hs]
    std::vector<tensor_t> mlp_down_w;  // SwiGLU 下降 [hs, di]

    // ---- KV-Cache ----
    // 每层两个缓存张量：[maxseq, nkvh, dh]，先按最大容量分配，
    // 用 slice 视图取"当前已写入的部分"。
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    size_t cache_len = 0;  // 当前缓存里有多少个历史 token

    // ========================================================================
    // 构造函数：接收 meta（权重由 chaosuanQwen2ModelWeights 提供，稍后填入）
    // ========================================================================
    Qwen2Model(const ChaosuanQwen2Meta &m, chaosuanDeviceType_t dev, int dev_id)
        : dtype(m.dtype), nlayer(m.nlayer), hs(m.hs), nh(m.nh), nkvh(m.nkvh), dh(m.dh),
          di(m.di), maxseq(m.maxseq), voc(m.voc), epsilon(m.epsilon), theta(m.theta),
          end_token(m.end_token), device(dev), device_id(dev_id) {
        // 先按最大容量分配 KV-Cache（每层 [maxseq, nkvh, dh]）。
        // 权重等 Python 填完 Weights 结构后，在第一次 infer 时惰性加载。
        for (size_t i = 0; i < nlayer; i++) {
            k_cache.push_back(chaosuan::Tensor::create({maxseq, nkvh, dh}, dtype, dev, dev_id));
            v_cache.push_back(chaosuan::Tensor::create({maxseq, nkvh, dh}, dtype, dev, dev_id));
        }
    }

    // ========================================================================
    // ensure_weights —— 惰性收编 Python 填好的权重（只在第一次 infer 前调用）
    // ========================================================================
    void ensure_weights() {
        if (weights_ready) return;

        // take：记录原始指针到释放清单，并取出共享 tensor 指针。
        auto take = [&](chaosuanTensor_t t) -> tensor_t {
            CHECK_ARGUMENT(t != nullptr, "Qwen2: a weight tensor is null (did you fill all weights?).");
            owned_weights.push_back(t);
            return t->tensor;
        };

        in_embed   = take(weights.in_embed);
        out_embed  = take(weights.out_embed);
        out_norm_w = take(weights.out_norm_w);

        for (size_t i = 0; i < nlayer; i++) {
            attn_norm_w.push_back(take(weights.attn_norm_w[i]));
            attn_q_w.push_back(take(weights.attn_q_w[i]));
            attn_q_b.push_back(take(weights.attn_q_b[i]));
            attn_k_w.push_back(take(weights.attn_k_w[i]));
            attn_k_b.push_back(take(weights.attn_k_b[i]));
            attn_v_w.push_back(take(weights.attn_v_w[i]));
            attn_v_b.push_back(take(weights.attn_v_b[i]));
            attn_o_w.push_back(take(weights.attn_o_w[i]));
            mlp_norm_w.push_back(take(weights.mlp_norm_w[i]));
            mlp_gate_w.push_back(take(weights.mlp_gate_w[i]));
            mlp_up_w.push_back(take(weights.mlp_up_w[i]));
            mlp_down_w.push_back(take(weights.mlp_down_w[i]));
        }

        weights_ready = true;
    }

    // ========================================================================
    // 析构函数：释放所有权重张量（C API 层的 ChaosuanTensor）
    // 注意去重：in_embed 与 out_embed 可能指向同一个指针。
    // ========================================================================
    ~Qwen2Model() {
        // 先去重：按指针地址排序后 unique。
        std::vector<chaosuanTensor_t> uniq = owned_weights;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        for (auto t : uniq) delete t;  // delete = 释放 ChaosuanTensor（内部 shared_ptr 自动管理 Tensor）
    }

    // ========================================================================
    // infer —— 对一批"新 token"做前向，返回 argmax 的下一 token
    // ------------------------------------------------------------------------
    // 关键设计：token_ids 是"新增"的 token，不是全部历史！
    //   - 第一次调用：传入整个 prompt（ntoken = prompt 长度），prefill；
    //   - 后续每次：传入上一个刚生成的 token（ntoken = 1），decode。
    // 新 token 的 k/v 前向完就写入缓存，下次直接复用。
    // ========================================================================
    int64_t infer(const int64_t *token_ids, size_t ntoken) {
        ensure_weights();  // 第一次推理前，先把 Python 填好的权重收编进来

        // ---- 1. 创建 index 张量（[ntoken]，I64），embedding 查表 ----
        auto index = chaosuan::Tensor::create({ntoken}, CHAOSUAN_DTYPE_I64, device, device_id);
        index->load(token_ids);  // 把 int64 数据拷进张量

        // hidden = embedding(index, in_embed)：形状 [ntoken, hs]
        auto hidden = chaosuan::Tensor::create({ntoken, hs}, dtype, device, device_id);
        chaosuan::ops::embedding(hidden, index, in_embed);

        // ---- 2. 逐层前向 ----
        // pos_ids：这批新 token 的绝对位置 = [cache_len, cache_len+1, ...]
        auto pos = make_pos_ids(cache_len, ntoken);
        auto pos_ids = chaosuan::Tensor::create({ntoken}, CHAOSUAN_DTYPE_I64, device, device_id);
        pos_ids->load(pos.data());

        for (size_t l = 0; l < nlayer; l++) {
            // ================= 注意力子层 =================
            // 2.1 归一化：h_norm = rms_norm(hidden)
            auto h_norm = chaosuan::Tensor::create({ntoken, hs}, dtype, device, device_id);
            chaosuan::ops::rms_norm(h_norm, hidden, attn_norm_w[l], epsilon);

            // 2.2 三个线性投影得到 q/k/v
            //   q: [ntoken, nh*dh]；k: [ntoken, nkvh*dh]；v: [ntoken, nkvh*dh]
            auto q = chaosuan::Tensor::create({ntoken, nh * dh}, dtype, device, device_id);
            chaosuan::ops::linear(q, h_norm, attn_q_w[l], attn_q_b[l]);
            auto k = chaosuan::Tensor::create({ntoken, nkvh * dh}, dtype, device, device_id);
            chaosuan::ops::linear(k, h_norm, attn_k_w[l], attn_k_b[l]);
            auto v = chaosuan::Tensor::create({ntoken, nkvh * dh}, dtype, device, device_id);
            chaosuan::ops::linear(v, h_norm, attn_v_w[l], attn_v_b[l]);

            // 2.3 重排成 3 维 [ntoken, nhead, dh]（view 是零拷贝视图，作业 1 的成果）
            auto q3 = q->view({ntoken, nh, dh});
            auto k3 = k->view({ntoken, nkvh, dh});
            auto v3 = v->view({ntoken, nkvh, dh});

            // 2.4 旋转位置编码（只对 q 和 k，v 不需要位置编码）
            auto qr = chaosuan::Tensor::create({ntoken, nh, dh}, dtype, device, device_id);
            chaosuan::ops::rope(qr, q3, pos_ids, theta);
            auto kr = chaosuan::Tensor::create({ntoken, nkvh, dh}, dtype, device, device_id);
            chaosuan::ops::rope(kr, k3, pos_ids, theta);

            // 2.5 把新的 k/v 写入 KV-Cache：
            //     用 slice 取缓存中 [cache_len, cache_len+ntoken) 这段，然后 load 拷贝。
            //     slice 返回的视图 data() 正好指向缓存里对应偏移，load 按元素数拷贝。
            auto k_w = k_cache[l]->slice(0, cache_len, cache_len + ntoken);
            auto v_w = v_cache[l]->slice(0, cache_len, cache_len + ntoken);
            // 【作业4 设备化】CPU 版直接 load（host→host 拷贝）。
            // GPU 版 kr/v3 与缓存都在显存里，必须用 memcpy D2D（设备→设备），
            // 不能走 load()（load 的语义固定是 H2D，host→设备）。
            if (device == CHAOSUAN_DEVICE_NVIDIA) {
                core::context().runtime().api()->memcpy_sync(
                    k_w->data(), kr->data(), k_w->numel() * k_w->elementSize(), CHAOSUAN_MEMCPY_D2D);
                core::context().runtime().api()->memcpy_sync(
                    v_w->data(), v3->data(), v_w->numel() * v_w->elementSize(), CHAOSUAN_MEMCPY_D2D);
            } else {
                k_w->load(kr->data());
                v_w->load(v3->data());
            }

            // 2.6 拼接后的完整 KV 视图：缓存 [0, cache_len+ntoken) 部分
            auto k_full = k_cache[l]->slice(0, 0, cache_len + ntoken);
            auto v_full = v_cache[l]->slice(0, 0, cache_len + ntoken);

            // 2.7 自注意力（算子内部会处理因果掩码 + GQA）
            auto attn = chaosuan::Tensor::create({ntoken, nh, dh}, dtype, device, device_id);
            float scale = 1.0f / std::sqrt(static_cast<float>(dh));
            chaosuan::ops::self_attention(attn, qr, k_full, v_full, scale);

            // 2.8 o 投影：把注意力输出映射回 hidden 维度，再加残差
            auto o = chaosuan::Tensor::create({ntoken, hs}, dtype, device, device_id);
            chaosuan::ops::linear(o, attn->view({ntoken, nh * dh}), attn_o_w[l], nullptr);
            // 残差连接：hidden = hidden + o（add 算子输出新张量）
            auto hidden2 = chaosuan::Tensor::create({ntoken, hs}, dtype, device, device_id);
            chaosuan::ops::add(hidden2, hidden, o);
            hidden = hidden2;

            // ================= MLP 子层 =================
            // 2.9 归一化
            auto h_norm2 = chaosuan::Tensor::create({ntoken, hs}, dtype, device, device_id);
            chaosuan::ops::rms_norm(h_norm2, hidden, mlp_norm_w[l], epsilon);

            // 2.10 SwiGLU：gate 和 up 两个投影 → 逐元素门控激活
            auto gate = chaosuan::Tensor::create({ntoken, di}, dtype, device, device_id);
            chaosuan::ops::linear(gate, h_norm2, mlp_gate_w[l], nullptr);
            auto up = chaosuan::Tensor::create({ntoken, di}, dtype, device, device_id);
            chaosuan::ops::linear(up, h_norm2, mlp_up_w[l], nullptr);
            auto glu = chaosuan::Tensor::create({ntoken, di}, dtype, device, device_id);
            chaosuan::ops::swiglu(glu, gate, up);

            // 2.11 下降投影 + 残差
            auto down = chaosuan::Tensor::create({ntoken, hs}, dtype, device, device_id);
            chaosuan::ops::linear(down, glu, mlp_down_w[l], nullptr);
            auto hidden3 = chaosuan::Tensor::create({ntoken, hs}, dtype, device, device_id);
            chaosuan::ops::add(hidden3, hidden, down);
            hidden = hidden3;
        }

        // ---- 3. 最后一层归一化 ----
        auto h_final = chaosuan::Tensor::create({ntoken, hs}, dtype, device, device_id);
        chaosuan::ops::rms_norm(h_final, hidden, out_norm_w, epsilon);

        // ---- 4. logits = h_final @ out_embed^T：形状 [ntoken, voc] ----
        auto logits = chaosuan::Tensor::create({ntoken, voc}, dtype, device, device_id);
        chaosuan::ops::linear(logits, h_final, out_embed, nullptr);

        // ---- 5. 取最后一个 token 的 logits 行做 argmax（贪心采样）----
        auto last_row = logits->slice(0, ntoken - 1, ntoken);  // [1, voc]
        auto last_1d = last_row->view({voc});                  // 拉平成 1D（argmax 要求 1D）
        auto max_idx = chaosuan::Tensor::create({1}, CHAOSUAN_DTYPE_I64, device, device_id);
        auto max_val = chaosuan::Tensor::create({1}, dtype, device, device_id);
        chaosuan::ops::argmax(max_idx, max_val, last_1d);

        // ---- 6. 更新缓存长度，返回生成的下一个 token ----
        cache_len += ntoken;
        // 【作业4 设备化】CPU 版：data() 就是 host 指针，直接解引用。
        // GPU 版：data() 是显存地址，CPU 不能直接读（会段错误），
        // 必须用 memcpy D2H（设备→host）拷回一个 int64 再返回。
        if (device == CHAOSUAN_DEVICE_NVIDIA) {
            int64_t next = 0;
            core::context().runtime().api()->memcpy_sync(
                &next, max_idx->data(), sizeof(int64_t), CHAOSUAN_MEMCPY_D2H);
            return next;
        }
        return *reinterpret_cast<int64_t *>(max_idx->data());
    }
};

} // namespace models
} // namespace chaosuan

// ============================================================================
// C API 实现（给 Python ctypes 调用）
// ============================================================================
__C {
    using chaosuan::models::Qwen2Model;

    // 创建模型：只保存超参数并分配 KV-Cache，返回不透明指针。
    // 权重稍后通过 chaosuanQwen2ModelWeights() 填入。
    struct ChaosuanQwen2Model *chaosuanQwen2ModelCreate(const ChaosuanQwen2Meta *meta,
                                                        chaosuanDeviceType_t device,
                                                        int *device_ids, int ndevice) {
        // 作业 3 只支持 CPU；作业 4 起支持 NVIDIA（RTX 4090 单卡）。
        // device_ids/ndevice：多卡分发参数，单卡作业直接用 device_id=0。
        (void)device_ids;
        (void)ndevice;
        CHECK_ARGUMENT(device == CHAOSUAN_DEVICE_CPU || device == CHAOSUAN_DEVICE_NVIDIA,
                       "Qwen2: only CPU/NVIDIA supported.");
        return reinterpret_cast<struct ChaosuanQwen2Model *>(new Qwen2Model(*meta, device, 0));
    }

    // 销毁模型：释放所有权重（去重后 delete）+ KV-Cache + 模型自身。
    void chaosuanQwen2ModelDestroy(struct ChaosuanQwen2Model *model) {
        delete reinterpret_cast<Qwen2Model *>(model);
    }

    // 获取权重结构指针：Python 侧往这个结构里填张量（所有权转移给模型）。
    struct ChaosuanQwen2Weights *chaosuanQwen2ModelWeights(struct ChaosuanQwen2Model *model) {
        return &reinterpret_cast<Qwen2Model *>(model)->weights;
    }

    // 推理：输入新 token（首次为整个 prompt，之后每次 1 个），返回下一 token。
    int64_t chaosuanQwen2ModelInfer(struct ChaosuanQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        return reinterpret_cast<Qwen2Model *>(model)->infer(token_ids, ntoken);
    }
}
