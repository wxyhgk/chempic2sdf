# Symbolic Candidate Family v0.4 预注册

日期：2026-08-15

## 裁决

v0.3 独立 completeness NO-GO 后，v0.4 采用：

```text
权威：root canonical objects + verifier-generated finite-domain CSP
可选：SAT witness、LRAT、ROBDD/SDD/d-DNNF 派生证书
展示：最多 128 个 materialized interpretation，仅调试
```

不用显式列表承担完整性。8 个独立二元歧义已有可执行探针验证：符号文档只有 8 个变量，满足赋值为 256，
显式 128 上限必溢出。

## 核心不变量

```text
Legal(root,I)
<-> exists total assignment a,
      Sat(CompileCSP(root),a)
      and CanonicalCompile(root,a)=Canonicalize(I)
```

`->` 是 completeness，`<-` 是 soundness。两向都必须证明。

producer 不得提供任意 CSP/CNF；独立 verifier 从 root 重建变量、精确 domain 和全部 hard constraints。否则
删除变量/domain value 或增加 unit clause就能再次制造 false singleton。

## 风险门

- root constructor 自身可能漏真实 PNG 候选，继续依赖公开 truth coverage；
- SAT model 与 semantic interpretation 可能多对一，必须消除 lane/slot/方向 symmetry；
- auxiliary variable 不得参与 semantic count；
- BDD canonical 只对固定变量序成立，且必须证明 `BDD <-> CSP`；
- UniqueGraph 需要 inequivalent-pair miter 的独立 UNSAT proof，不属于 v0.4 第一版；
- solver timeout、unknown 和 proof checker failure只能 Reject。

## 实施顺序

1. 冻结 root object constructor 的有限微场景；
2. 冻结 closed finite-domain constraint AST；
3. 做 root objects、变量、约束、proof bytes 容量 census；
4. 证明 CSP sound/complete 与 canonical semantic projection；
5. 构造 256 解、domain 删除、unit clause、symmetry 和 cross-component fixture；
6. 独立红队 GO 后再写 production schema；
7. public proposer、UniqueGraph、SDF、训练继续关闭。

详细协议草案见 `Data/benchmarks/carrier_candidate_family_v0_4/README.md`。
