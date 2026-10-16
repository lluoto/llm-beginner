# AI Tutor 评审报告 · 任务二 mini-GPT

> 评审对象：`task-2-mini-gpt/`（`src/tokenizer.py`、`src/rope.py`、`src/attention.py`、`src/model.py`、`train.py`、`sampling.py`）
> 复现环境：`data/simple`（唐诗档），`d_model=8, n_head=1, d_ff=32, vocab=256, max_len=256`
> 自检结果（`eval/result.json`）：3/3 通过。

---

## 概览

整体水平：**能跑通、三项自检全过，工程骨架完整**（BPE→RoPE→KV cache→训练→采样全链路手写，含 tensorboard 日志与早停），但存在 **3 个致命设计缺陷** 与若干接口/训练规范问题：

1. **BPE 合并表被词表截断彻底作废** —— 学了 100 条 merge，但 `vocab_size=256` 连「256 字节 + 4 特殊符」都装不下，merge 结果全部被 `vocab[:256]` 丢掉，分词退化为**纯字节级**（中文 1 字 = 3 token，压缩率 0.34 < 1），比字符级还差。
2. **`vocab_size` 与 `max_len` 被同一个数 `256` 复用** —— 词表大小、上下文长度、logits 维度三处耦合，换语料必崩。
3. **训练"验证"其实是训练集** —— `train.py` 在 `train_set` 上算 PPL 当早停指标，从未真正读 dev split，M4 的泛化监控名不副实。

关键问题数（需修复）：**必检 6 项中 2 项需修**（tokenizer 的 merge 被截断、训练用了训练集当验证），KV cache / 采样为「功能正确但有缺口 / 实验配置冗余」，RoPE / causal attention 通过。加分项 4 项命中 1 项。

> 勘误（v2）：上一版把「`temperature=0` 时忽略 top-k/top-p」列为逻辑缺陷，这是**错的**。`temp=0` 的定义就是 greedy（`softmax(logits/T)` 在 `T→0` 的极限为 one-hot），而 top-k 必含 argmax、top-p 的 `keep[...,0]=1` 也必保 argmax，故二者对结果无影响——**忽略它们才是正确语义**。已改写第 6 项；同时澄清训练项中"验证集泄漏"的确切含义。

---

## 参数量梯度实验（你关注的 11 / 62 / 125 层）

> 说明：`train.py` 用 `for layer_para in [11,62,125]` 做参数扫描，**层数 = 参数量的旋钮**。三档结果如下（参数量为 `model.parameters()`，不含 RoPE buffer；dev PPL 由我用 `data/simple/dev.txt` 前 4096 token、按 256 分块复算；train PPL 为训练日志末值）：

| 档位 | 层数 N | 参数量 | 每层参数 | 固定参数(embed+lm_head) | dev PPL | train PPL(日志末值) | 训练稳定性 |
|---|---|---|---|---|---|---|---|
| 小 | 11 | **13,944** (≈1.4e4) | 872 | 4,352 | **15.70** | 14.74 | 平稳 |
| 中 | 62 | **58,416** (≈5.8e4) | 872 | 4,352 | **15.28** | 14.55 | 起始 `inf` 尖峰 |
| 大 | 125 | **113,352** (≈1.1e5) | 872 | 4,352 | **15.91** | 14.91 | 起始 `inf` + 中途多次尖峰 |

参数公式：`4352 + N×872`（attn 288 + FFN 552 + LN 32 = 872/层），三档比例 1 : 4.19 : 8.13。

**结论（重要）**：
- **实际参数量比 S1 目标低约 3 个数量级**。`train.py:84` 的注释 `#62 *256*1024 =50 M 12 * 256*1024=10M` 是按 `d_model=256, d_ff≈1024`（≈80 万参数/层）估算的 10M/50M/100M，但实际跑的是 `d_model=8, d_ff=32`（872 参数/层），所以得到的是 **1.4e4 / 5.8e4 / 1.1e5**，不是 10M/50M/100M。若要做真正的参数量扫描，必须调 `-dmodel`（如 128/256/512）而不是只加层数。
- **困惑度不是随参数量单调下降**：11→62 层（参数 ×4.2）dev PPL 仅从 15.70 降到 15.28（−0.42），62→125 层（参数 ×1.9）反而升到 15.91（+0.63）。在 43KB 唐诗上，**深度堆叠已无收益甚至轻微过拟合/优化变差**，这是「数据量 << 模型容量」的典型信号。
- **深而窄的堆叠还带来训练不稳定**：62/125 层在前若干 step 出现 `inf`，125 层在中途（step≈63–67）出现 `22304 / 141638` 的 PPL 尖峰；11 层则平滑。这与下面「norm 用法不规范 + warmup 相对偏短」相互印证。
- 三档都远超阈值（<50），自检通过；但**训练日志里的 `ppl` 是训练集 PPL**（见必检项「训练」），所以"最优"早停点的泛化性没有被真正监控。

---

## 必检项

### 1. BPE tokenizer（`src/tokenizer.py`）

- 状态：**需要修复**（自检 `tokenizer_roundtrip` 通过，但 BPE 事实上未生效）
- 现状：
  ```python
  # build_bpe_vocab
  basal=[f'byte_{i}' for i in range(256)]
  vocab=self.bpe_merge(text)               # 返回 merge 后的 token 字符串
  special_token=['<pad>','<unk>','<bos>','<eos>']
  vocab=special_token+basal+vocab          # 4 + 256 + N
  vocab=vocab[:vocab_size]                 # vocab_size = args.max_len = 256
  ```
- 问题：
  - `4 + 256 = 260 > 256`，`vocab[:256]` 把 **全部 merge token 截掉**，只剩 `<pad> <unk> <bos> <eos>` + `byte_0..byte_251`（连 `byte_252..255` 也没了）。实测 **256 词表 / 100 条 merge / 词表里 merge token 数 = 0**。
  - 于是 `encode` 里 `if key in self.word2idx` 永远对 merge 结果判 False，走 `for b in token: byte_{b}` 逐字节 fallback。实测 `床前明月光`（5 字）→ **15 个 token**（3 字节/字），全语料压缩率 **0.344（<1，即膨胀 3 倍）**。BPE 的核心价值（把常见子串合成 1 个 token）完全没兑现。
  - `merge 过程本身是真实现的**（统计相邻对频率 → 取最高频合并 → 重建统计），没有用 tiktoken/sentencepiece，这点合格；`<bos>/<eos>/<pad>/<unk>` 也预留了。但**等长 merge 顺序在 `encode` 时按 dict 插入序而非 rank 序**，严格说与训练时的 merge 顺序语义有偏差（当前因 merge 全被丢弃而未暴露）。
  - `vocab_size` 与 `max_len` 共用 `256`（`train.py:99` `BPETokenizer(128, args.max_len, ...)`），且未实现 README 要求的 `vocab_size` 属性（`hasattr(tok,'vocab_size')==False`）。词表里有 `byte_252..255` 缺失，遇到 0xFC–0xFF 字节会映射到 `<unk>`，`decode(errors='replace')` 又把失败悄悄吞掉。
- 修复建议：
  ```python
  # tokenizer.py：让 merge 真正进入词表，并解耦 vocab_size / max_len
  special_token = ['<pad>','<unk>','<bos>','<eos>']
  basal = [f'byte_{i}' for i in range(256)]
  n_merge = max(0, vocab_size - len(special_token) - len(basal))   # 只保留能装下的 merge
  merges = self.bpe_merge(text, process=n_merge)
  vocab = special_token + basal + merges
  vocab = vocab[:vocab_size]
  self.vocab_size = len(vocab)                # 暴露 README 要求的属性

  # train.py：词表大小单独给参数，别再借用 max_len
  parser.add_argument('--vocab_size', type=int, default=4096)
  BPE = BPETokenizer(128, args.vocab_size, args.train_dir)
  ...
  MiniGPT(..., vocab_len=len(BPE.word2idx))   # 而不是 args.max_len
  ```
  同时把 `build_bpe_vocab` 里的 `str.decode('utf-8')` 改为 `errors` 容错，避免纯字节序列边界崩溃。
- **`vocab_size` 到底该设多少（结论）**：
  - 词表结构是 `4 特殊符 + 256 byte + N merge`，所以 **硬下限 = 260**；低于 260（当前的 256）就会把 merge 全截掉。
  - 有效 merge 数 = `vocab_size − 260`，据此挑选。
  - 本任务 **43KB 唐诗**（去重汉字约几千）推荐 **2048 或 4096**；merge 太多会大量"只用过一两次"、浪费 embedding。换 TinyStories/SkyPile 可提到 8k–16k，中文大语料 32k 级。
  - 判据不是"越大越好"，而是 **压缩率 `chars/tokens` 明显 > 1**（中文理想 1.2–3）且 `<unk>` 占比极低。当前压缩率 **0.344 < 1**（1 字 = 3 token，反而膨胀），即有效 merge = 0。
  - 与模型的耦合：`MiniGPT(vocab_len=...)` 必须等于 tokenizer **实际**词表长度 `len(word2idx)`，既不是 256 也不是 `max_len`。

### 2. RoPE（`src/rope.py`）

- 状态：**通过**
- 现状：`freq` 在 `__init__` 用 `register_buffer` 建一次表，`forward` 里复用；只对 Q/K 调用（`attention.py` 里 `self.Q_rope(Q)` / `self.K_rope(K)`，V 未加）；增量解码用 `offset=cur_pos` 接续历史位置。
- 问题：基本正确。两点可改进：
  1. 采用「前后折半」配对（`x1=x[...,:d//2]`，`x2=x[...,d//2:]`，GPT-NeoX 风格）而非相邻两维配对（HF 风格）；**只要训练/推理一致就正确**，这里是一致的。但 `d//2=4` 且 `freq` 长度 `head_dim/2` 与之匹配，正确。
  2. `freqs = torch.outer(t, self.freq)` 每次 forward 重算（未缓存到 max_seq_len），长上下文时有点浪费；可预建 `cos/sin` 表并按需切片。
- 修复建议：功能无需改；如需性能，预先在 `__init__` 建 `max_seq_len` 长度的 cos/sin 缓存。

### 3. Causal attention（`src/attention.py`）

- 状态：**通过**
- 现状：
  ```python
  casual_mask=torch.triu(torch.ones(L,L),diagonal=1).bool()   # 上三角 True = 屏蔽
  mask=casual_mask|mask                                        # 与 padding mask 合并
  scale_mat=torch.masked_fill(scale_mat,mask,float('-inf'))
  soft_mat=torch.softmax(scale_mat,dim=-1)
  ```
- 问题：`-inf` + 上三角写法正确；`softmax` 由 PyTorch 内部减 max，数值稳定（全程 fp32，无 half 溢出风险）。唯一隐患：若某 query 行**全部**被 mask（全 padding 行），softmax 会出现 `-inf` 相加 → NaN；当前训练用滑窗无 padding、causal 保证每行至少 1 个可见位置，所以未触发。
- 修复建议：可选加 `scale_mat.masked_fill_(全部可见, 0)` 兜底；非必需。

### 4. KV cache（`src/attention.py` + `src/model.py`）

- 状态：**通过（等价性），但缺上下文上限，且接口未暴露**
- 现状：
  ```python
  if K_cache is not None:
      cur_pos=K_cache.shape[2]
      Q=self.Q_rope(Q,offset=cur_pos)
      K=torch.cat([K_cache,K],dim=2)   # 在序列维 dim=2 拼接，正确
      V=torch.cat([V_cache,V],dim=2)
  Q_last=Q[:,:,-1:,:]                  # 增量步只 forward 最后一个 token
  ```
  自检 `kv_cache_equivalence`：`max_abs_diff=2.86e-06 < 1e-4`，通过。
- 问题：
  - 拼接维度正确（在 K/V 的 T 维 `dim=2`），RoPE 增量 offset 正确，推理只算最后词元——这三点都对。
  - **cache 长度无上限**，长生成会一直增长，没有滑动窗口 / `max_cache_len` 截断。
  - 模型**未暴露 `block_size` / `max_seq_len`**（实测 `hasattr` 均为 False，只有 `self.max_len=vocab_len`）。自检因此只能走默认 256；这恰好等于训练长度，属于"碰巧对"，一旦改结构就会失配。
- 修复建议：
  ```python
  # model.py
  self.block_size = max_len                 # 显式暴露，供自检/推理切窗
  # 或 self.max_seq_len = max_len

  # attention.py：cache 超长时截断最旧的部分（滑动窗口）
  if K.shape[2] > self.max_cache_len:
      K = K[:, :, -self.max_cache_len:, :]
      V = V[:, :, -self.max_cache_len:, :]
  ```

### 5. 训练（`train.py`）

- 状态：**需要修复**
- 现状：
  ```python
  warmup=args.cosine_temp*total_step
  def lambda_para(step):
      if step<warmup: return step/warmup
      progress=(step-warmup)/max(1,total_step-warmup)
      return 0.5*(1+math.cos(math.pi*progress))
  ...
  clip_grad_norm_(model.parameters(),max_norm=1.0)
  ```
- 问题：
  - **验证集用错对象——用训练集当验证（真正的 bug）**。第 180 行 `for batch in train_set:`：用 `train_set`（不是验证集）算 `average_loss/average_ppl`，并据此早停、存 `best_*.pt`。用训练集早停等价于永远朝着"记住训练集"的方向选模型，泛化性无从监控，README/M4 要求"留出 dev split 监控困惑度"未被满足（`data/simple/dev.txt` 只有 `eval/run.py` 用了）。
  - **关于"泄漏"的澄清**：用验证集做监控/早停本身**不是泄漏**（那是验证集的用途）；泄漏有两种：(a) 训练时把梯度更新到 dev 上；(b) **同一份 dev 既用于模型选择、又当最终 held-out 指标**——此时最终分数会偏乐观。仓库有 `data/train.txt / dev.txt / test.txt`，规范做法二选一：从 `train.txt` 里切一份 validation 做早停、把 `data/dev.txt` 留给 `eval/run.py` 的自检；或直接用 `data/dev.txt` 早停、最终汇报/评估改用 `data/test.txt`。当前代码两种都没做（因为它压根没验证集）。
  - 已具备且正确的部分：warmup + cosine（`LambdaLR`）、`clip_grad_norm_(max_norm=1.0)`、tensorboard 记 `ppl` 曲线、基于相对改善 + patience 的早停。
  - 深层模型的 `inf`/尖峰（见参数量小节）与下面这个**非标准 norm 用法**有关：
    ```python
    # src/model.py decoderlayer.forward
    out,_=self.mask_att(self.norm1(x),...)     # 注意力分支：norm 作用在输入（pre-norm）
    x=x+self.drop(out)
    x=x+self.norm2(self.fnn(self.drop(x)))     # FFN 分支：norm 作用在输出（post-norm）
    ```
    两个分支一个 pre-norm 一个 post-norm，语义不统一，深堆叠时更易不稳定。
  - 收尾的参数公式打印用的是 `args.layer_number`（默认 12）而非循环变量 `layer_para`，输出会误导（`train.py:219`）。
- 修复建议：
  ```python
  dev_set = DataLoader(read_dataset(args.dev_dir, BPE, args.max_len), batch_size=args.batch_size)
  ...
  model.eval()
  total_loss=0
  with torch.no_grad():
      for batch in dev_set:                       # ← 用 dev，而不是 train_set
          ...
  # 统一为 pre-norm：
  x = x + self.drop(out)
  x = x + self.drop(self.fnn(self.norm2(x)))
  # 打印修正：args.layer_number → layer_para
  ```

### 6. 采样（`src/model.py generate`）

- 状态：**通过（temp=0 语义正确）**；仅实验配置冗余 + 一个未触发的潜在组合 bug
- 现状：
  ```python
  if temperature==0:
      next_id=next_logits.argmax(dim=-1,keepdim=True)   # greedy
  else:
      next_logits=next_logits/temperature
      if top_k: ...
      if top_p: ...
      prob=torch.softmax(next_logits,dim=-1)
      next_id=torch.multinomial(prob,1)
      if top_k: next_id=topk_indice.gather(-1,next_id)
      if top_p: next_id=topp_idx.gather(-1,next_id)
  ```
- 问题（**已勘误**）：
  - `temperature=0` 退化为 greedy 是**定义上正确**的（`softmax(logits/T)` 在 `T→0` 时收敛到 one-hot）。此时 top-k 必含 argmax、top-p 因 `keep[...,0]=1` 也必保 argmax，**top-k/top-p 对结果没有任何影响**——所以"`temp==0` 时忽略它们"**不是 bug**，反而避免了无意义的过滤。`temperature=0` 也**不会**触发 0/0（除零只在 `logits/temperature` 时发生，而该分支根本不做除法）。
  - **真正可提的是实验配置冗余**：`sampling.py` 里 group 0（`temp=0`）、group 3（`temp=0, top_k=30`）、group 4（`temp=0, top_p=0.9`）三者输出**必然完全相同**（都退化成 greedy），放在同一张对比表里没有区分度；应把 3/4 改成 `temp>0` 才构成有效消融。
  - **潜在 bug（当前未触发）**：仅当 `top_k` 与 `top_p` **同时开启**时，两次 `gather` 的索引空间不一致——`topk_indice` 是 vocab 空间的索引，而 `topp_idx` 是对 top-k 之后长度为 k 的向量排序得到的**k 空间**索引，`topp_idx.gather(-1, next_id)` 会越界/错位。`sampling.py` 从不组合二者，所以没暴露。
  - top-p 的其余顺序是对的：降序 → softmax → `cumsum` → `keep` 右移（保证跨过阈值的第一个也保留）→ `where(-inf)` → **重新 softmax 归一化** → `topp_idx.gather` 映射回原索引。
  - 生成除首步外**未再屏蔽 `<pad>`**（首步 `next_logits[:,pad]=-inf`）；无 repetition penalty。batch 在张量层面可用（`next_id` 形状 `[B,1]`），但 `sampling.py` 只打印 `output[0]`。
- 修复建议：
  - 若保持单一 `temperature=0 → greedy` 语义（推荐），**代码无需改**；只把 `sampling.py` 的 group 3/4 换成 `temp>0` 的组合即可。
  - 若确实想支持"`temp==0` 且同时 top-k+top-p"，需让过滤与采样在同一索引空间完成，例如先在原 vocab 上应用 top-p 得到 mask、再套 top-k、最后统一做一次 argmax/采样，只 remap 一次：
  ```python
  if temperature != 0:
      next_logits = next_logits / temperature
  if top_k:                       # 在 vocab 空间直接屏蔽，不换索引空间
      kth = torch.topk(next_logits, top_k, dim=-1).values[..., -1:]
      next_logits = next_logits.masked_fill(next_logits < kth, float('-inf'))
  if top_p:                       # 排序/累积/截断后，用 scatter 把 keep 还原成 vocab 空间的 mask
      sorted_logits, sorted_idx = torch.sort(next_logits, dim=-1, descending=True)
      keep = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) <= top_p
      keep[..., 1:] |= keep[..., :-1]
      keep[..., 0] = True
      mask = torch.zeros_like(next_logits, dtype=torch.bool)
      mask.scatter_(-1, sorted_idx, keep)
      next_logits = next_logits.masked_fill(~mask, float('-inf'))
  if temperature == 0:
      next_id = next_logits.argmax(dim=-1, keepdim=True)          # greedy
  else:
      next_id = torch.multinomial(torch.softmax(next_logits, -1), 1)  # 重新归一化后采样
  ```
  要点：全程留在 **vocab 空间**（只用 mask，不做 `gather` 索引映射），就不会再出现 `topk_indice`（vocab 空间）与 `topp_idx`（k 空间）混用导致的越界/错位。

---

## 加分项观察

1. **weight tying（未做）**：`TransformerBlock` 里 `output_embedding` 与 `decode_linear` 是两套独立参数，`BPETokenizer` 里**还有第三个 embedding**且从未被使用。建议把 `output_embedding.weight` 与 `decode_linear.weight` 共享（`self.decode_linear.weight = self.output_embedding.weight`），对 256 词表这种小模型能省近 1/4 参数，且是标准做法。
2. **SwiGLU（未做）**：FFN 是 `Linear→ReLU→Linear`，可换 `SwiGLU`（`Linear→SiLU(gate)*up→Linear`）。
3. **tensorboard（已做）**：`SummaryWriter` 记了 `ppl` scalar 与逐参数 grad histogram，`add_hparams` 也接了。**但 grad histogram 每步都写导致 event 文件巨大**（单 run 目录 0.7–1.7 GB，单个 tfevents 达 145 MB），建议降频或只记范数。
4. **batch generation（部分）**：`generate` 的张量运算支持 batch，但 `sampling.py` 未真正批量输出对比。

---

## 优先级排序（actionable）

1. **修 tokenizer 词表截断（P0）**：`vocab=special+basal+merges` 后不要再被 `vocab_size` 砍光 merge；把 `vocab_size`（如 4096）与 `max_len`（256）拆成两个参数，并暴露 `vocab_size` 属性。这是"手写 BPE"真正的验收点，当前等于没生效。
2. **修训练验证指标（P0）**：早停/存 best 改在**验证集**上算 PPL（当前用的是 `train_set`）。为避免"验证/评估同源"，建议从 `train.txt` 切一份 validation 做早停、保留 `data/dev.txt` 给 `eval/run.py`；或直接用 `dev.txt` 早停、最终汇报改看 `test.txt`。
3. **暴露 `block_size` / `max_seq_len` 并给 KV cache 加上限（P1）**：让自检按真实上下文切窗；生成时滑动窗口截断 cache，防止无限增长。
4. **整理 `sampling.py` 实验分组 + 修潜在组合 bug（P1）**：`temp=0` 的 group 0/3/4 输出相同（greedy），把 3/4 改成 `temp>0` 才有对比意义；若将来同时用 top-k+top-p，需统一索引空间（见必检项 6 的修复代码）。
5. **参数量扫描要"扫对维度"（P2）**：11/62/125 层得到的是 1.4e4/5.8e4/1.1e5，而非 10M/50M/100M。若要复现 S1，请用 `-dmodel`（+`-dff`）做宽度扫描（如 d_model=128/256/512），并补齐真正 dev PPL 才能画出参数量-困惑度曲线。当前数据已显示：唐诗档上深度堆叠 11→62→125 收益递减并转负（15.70→15.28→15.91）。
6. **统一 norm 用法并加 weight tying（P2，附带加分）**：FFN 分支改为标准 pre-norm，deep stack 的 `inf` 尖峰应会缓解；`embedding` 与 `lm_head` 共享权重。

---

### 附：接口对照（README「实现约定」）

| 要求 | 现状 | 结论 |
|---|---|---|
| `BPETokenizer.encode/decode` | 有 | ✅ |
| `BPETokenizer.vocab_size` | **无该属性**（只有 `vocab` / `word2idx`） | ⚠️ |
| `BPETokenizer.from_pretrained(path)` | 有（第二参数 `embedding_dim` 有默认值，兼容自检） | ✅ |
| `MiniGPT.forward(ids, kv_cache, return_cache)` | 有 | ✅ |
| `MiniGPT.generate(prompt_ids, max_new_tokens, top_k, top_p, temperature)` | 有，但签名是 `generate(prompt_ids, temperature, top_k, top_p, ..., max_len)`，**无 `max_new_tokens`** | ⚠️ |
| `MiniGPT.block_size` / `max_seq_len` | **均无**（只有 `max_len`） | ⚠️ |
| `load_for_eval(ckpt_path)` | 有，但**硬编码** `d_model=8,n_head=1,N=12,d_ff=32`，无法加载 `best_62/best_125` | ⚠️ |
| `ckpt/tokenizer.json`、`ckpt/best.pt` | 存在 | ✅ |

---

## 附录 A · 生成任务的数据划分与「错位」target

### A.1 「错位」本身没有问题，它是 LM 的标准构造

```python
input, target = batch[:, :-1], batch[:, 1:]
```
这是 next-token prediction 的**标准 teacher-forcing 错位**，正确、不是问题源。

### A.2 真正会造成"隐性泄漏"的是**滑窗重叠 + 从同一条 token 流切分**

`train.py` 的 `read_dataset` 用 `stride = max_len//4`（`dataset=='simple'`）滑窗：

```python
for i in range(0, len(input_tensor)-self.max_len, stride):
    windows = input_tensor[i:i+self.max_len]
```
相邻窗口**重叠 3/4 的 token**。此时若你从 `self.block` 列表里直接切 train/val（例如 `block[:k]` / `block[k:]`），**相邻窗口会跨训练/验证边界共享 3/4 的内容**，验证 PPL 会被乐观污染——这才是"错位制造测试集"的隐患所在，而不是 `[:-1]` 那个位移。

**正确做法**：先按 **token 流顺序**（或按文档/诗句边界）切成互不重叠的几段，再在每段内部各自滑窗：
```python
n = len(ids)
cut = int(n * 0.9)
train_ids, val_ids = ids[:cut], ids[cut:]          # 先切流，窗口不跨界
# 再分别 build windows，val 可以不用 overlap 或 stride=max_len（避免自相关）
```
（更稳的是按 `\n\n` 段落切，避免把一首诗劈成两半，也避免验证集里出现训练集的原句。）

### A.3 要不要"另外切测试集"？——取决于你是否用 dev 做模型选择

- **生成任务的评估指标就是 teacher-forced PPL**（在 held-out 文本上算），这点和分类任务一样，**两分（train/dev）就够**，不强制要 test。
- 但要注意 **"验证集/最终指标同源"**：
  - `eval/run.py` 的 `perplexity_on_dev` 用 `data/simple/dev.txt` 当**最终评分**；
  - 如果 `train.py` 又拿同一份 `dev.txt` 去早停/调参，那 dev 就"既是选择集又是最终指标"，分数偏乐观。
  - 两种干净方案（二选一）：
    1. 从 `train.txt` 切 validation 做早停，**把 `dev.txt` 完整留给自检**（推荐，因为评分脚本固定读 dev）；
    2. 用 `dev.txt` 早停，额外留 `test.txt`（仓库已有 `data/test.txt`）做最终汇报。
- 结论：**不需要再"额外"造第三个集合**，需要的只是"别让同一份 dev 同时承担选择与评分"。

### A.4 附带的小问题

- `range(0, len-max_len, stride)` 会**丢掉末尾不足一个窗口、且最后一个完整窗口**的数据（`len-max_len` 处的窗口没取到），可用 `range(0, len-max_len+1, stride)` 补上。
- 滑窗重叠会让**训练集远大于语料本身**（43KB 语料在 `stride=64, max_len=256` 下被放大数倍），配合小语料更容易记住原文——这解释了参数量 11→62→125 层时 PPL 反而变差的现象。

---

## 附录 B · 为什么 TensorBoard 里看不到训练超参数（已核查）

**结论：hparams 其实写进去了**，不是"没记录"。我直接解析了 event 文件，确认包含：

```
_hparams_/experiment          # 12 个键：dimention_of_model, dimention_of_FFN, number_head,
                              # layer_number, learning_rate, dropout_rate, seed,
                              # cosine temp, relative_rate, patience, weight_decay + 指标 ppl
_hparams_/session_start_info  # 各键的数值
_hparams_/session_end_info
ppl                          # 指标（best_ppl）
```
所以"不可见"是**可见性 / 加载问题**，根因按可能性排序：

### 原因 1（结构）· `add_hparams(run_name=...)` 把 hparams 写进了**子目录**
```python
runtime_writer = SummaryWriter(f'runs/{dataset}/seed_{current_seed}_layer_{layer_para}')
...
runtime_writer.add_hparams(hparams, {'ppl': best_ppl}, run_name=f'seed_{current_seed}')
```
PyTorch 的 `add_hparams` **必然**新建一个子 writer：`<log_dir>/<run_name>/`。实测目录结构正是
`runs/simple/seed_1111111/seed_1111111/events...`（hparams 在这里），而训练曲线在
`runs/simple/seed_1111111/events...`（父目录）。
**TensorBoard 里 run = 目录**，所以 hparams 属于**另一个（嵌套的）run**。你在父 run 上看，或 run 选择器把嵌套 run 过滤掉，就"看不见"。
- 修：`tensorboard --logdir runs` 后，在左侧选中 `.../seed_1111111` 这个**子 run**，再切到 **HPARAMS 标签页**（不是 Scalars）。

### 原因 2（主因）· 巨型 event 文件拖垮 TensorBoard 加载
训练每 10% step 对**每个参数**写 `add_histogram`：
```python
for name, param in model.named_parameters():
    runtime_writer.add_histogram(f'grad/{name}', param.grad, accumulate_step)
```
实测后果：**单个 tfevents 144 MB，单 run 目录 0.7–1.7 GB**。我用 `tensorboard --logdir runs/simple/seed_1111111 --inspect` 尝试读取该 run，**超过 2 分钟仍未完成并被中断**；而只读那个 866 字节的嵌套 hparams 目录则是秒出。也就是说 TB 在解析这些巨型文件时卡住/超时，**HParams 后端拿不到数据，页面表现为空白**。
- 修：把逐参数 histogram 换成低频/只看范数，或直接去掉：
  ```python
  # 只记全局梯度范数，别记每个参数的直方图
  total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  runtime_writer.add_scalar('grad_norm', total_norm, accumulate_step)
  # 或者 add_histogram(..., accumulate_step) 改成每隔很多 step 且只记录少数关键层
  ```
  清理历史巨文件后再启动 TB：`rm -rf runs/simple/seed_1111111`（或只保留嵌套 hparams 目录）。

### 原因 3 · 多个 session 同名写进同一目录
`run_name=f'seed_{current_seed}'` 在固定 `--seed`（如 `-s 1111111`）时**每次迭代都是同名**，于是 12 次调用往同一个子目录塞了 12 个 866B 的 hparams event 文件（我数到 12 个）。多个 `_hparams_/experiment` 同时存在会让 HParams 插件只显示一个 session / 或相互覆盖。
- 修：让 `run_name` 带上层数，保证唯一：`run_name=f'seed_{current_seed}_layer_{layer_para}'`。

### 原因 4 · 看错标签页
超参数显示在 **HPARAMS** 标签页（表格：列=数值型超参，行=session），**不在 Scalars**。字符串型超参（`'seed': '1111111 simple'`、`'cosine temp'`）只会作为离散列/分组出现，容易以为"没记录"。另外 `TensorFlow installation not found - running with reduced feature set` 不影响 PyTorch 写的 hparams。

### 一句话修复清单
1. `run_name` 唯一化（带 layer_para）；2. 删掉逐参数 histogram（这是卡顿主因）；3. `tensorboard --logdir runs` 指向父级、选中嵌套子 run、切 **HPARAMS** 页；4. 清掉历史 GB 级 event 文件。
