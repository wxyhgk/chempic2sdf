import ChemGraph.RasterEvidence

namespace ChemGraph

/-!
Hidden Challenge v0.4 只形式化 generator-first 协议的逻辑接口。Lean 无法自行
读取 Git 时间、证明 seed 未泄漏、验证密码学抗碰撞，也无法证明某个版本值确实是
代码内容的 hash；这些事实必须由外部 commitment 和独立审计提供。
-/
def CommitmentBinds {Version Commitment : Type}
    (commit : Version -> Commitment) (version : Version)
    (published : Commitment) : Prop :=
  commit version = published

def CommitmentUniquelyBinds {Version Commitment : Type}
    (commit : Version -> Commitment) : Prop :=
  Function.Injective commit

/-!
`generatorBeforeExtractor` 是外部冻结顺序见证。这里证明的是：在 commitment
单射假设下，发布后不能用另一 GeneratorVersion 替换；并不证明现实时间顺序。
-/
structure GeneratorFirstFreeze
    {GeneratorVersion ExtractorVersion Commitment : Type}
    (commitGenerator : GeneratorVersion -> Commitment)
    (before : GeneratorVersion -> ExtractorVersion -> Prop)
    (generatorVersion : GeneratorVersion)
    (extractorVersion : ExtractorVersion)
    (publishedCommitment : Commitment) : Prop where
  generatorBound : CommitmentBinds
    commitGenerator generatorVersion publishedCommitment
  commitmentUnique : CommitmentUniquelyBinds commitGenerator
  generatorBeforeExtractor : before generatorVersion extractorVersion

theorem committed_generator_version_cannot_be_substituted
    {GeneratorVersion ExtractorVersion Commitment : Type}
    (commitGenerator : GeneratorVersion -> Commitment)
    (before : GeneratorVersion -> ExtractorVersion -> Prop)
    (generatorVersion alternative : GeneratorVersion)
    (extractorVersion : ExtractorVersion)
    (publishedCommitment : Commitment)
    (freeze : GeneratorFirstFreeze commitGenerator before
      generatorVersion extractorVersion publishedCommitment)
    (alternativeBound : CommitmentBinds
      commitGenerator alternative publishedCommitment) :
    alternative = generatorVersion := by
  apply freeze.commitmentUnique
  exact alternativeBound.trans freeze.generatorBound.symm

/-!
`EvidenceCore` 保持为任意类型参数，不要求 lane、terminal 或文字 mask 等具体字段。
充分性只由“经 Evidence 枚举出的候选集合与目标候选集合一致”定义。
-/
def SufficientForCandidates {Raster Evidence Graph : Type}
    (extract : Raster -> Evidence)
    (enumerate : Evidence -> CandidateSet Graph)
    (oracleCandidates : Raster -> CandidateSet Graph) : Prop :=
  forall raster graph,
    enumerate (extract raster) graph ↔ oracleCandidates raster graph

def CandidateOracleExactOnDomain
    {Scene Raster Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (meaning : Scene -> Graph)
    (oracleCandidates : Raster -> CandidateSet Graph) : Prop :=
  forall raster graph,
    oracleCandidates raster graph ↔
      GraphSupportedAt valid render meaning raster graph

def EvidenceExactOnDomain
    {Scene Raster Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (meaning : Scene -> Graph)
    (extract : Raster -> Evidence)
    (enumerate : Evidence -> CandidateSet Graph) : Prop :=
  forall raster graph,
    enumerate (extract raster) graph ↔
      GraphSupportedAt valid render meaning raster graph

theorem sufficient_evidence_and_exact_oracle_imply_exact_on_domain
    {Scene Raster Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (meaning : Scene -> Graph)
    (extract : Raster -> Evidence)
    (enumerate : Evidence -> CandidateSet Graph)
    (oracleCandidates : Raster -> CandidateSet Graph)
    (sufficient : SufficientForCandidates extract enumerate oracleCandidates)
    (oracleExact : CandidateOracleExactOnDomain
      valid render meaning oracleCandidates) :
    EvidenceExactOnDomain valid render meaning extract enumerate := by
  intro raster graph
  exact (sufficient raster graph).trans (oracleExact raster graph)

def HiddenStrictSampleUpTo {Sample Graph : Type}
    (equivalent : Graph -> Graph -> Prop)
    (candidates : Sample -> CandidateSet Graph) (sample : Sample) : Prop :=
  Exists fun graph =>
    candidates sample graph ∧
      forall other, candidates sample other -> equivalent other graph

def HiddenAmbiguousSampleUpTo {Sample Graph : Type}
    (equivalent : Graph -> Graph -> Prop)
    (candidates : Sample -> CandidateSet Graph) (sample : Sample) : Prop :=
  Exists fun left => Exists fun right =>
    candidates sample left ∧ candidates sample right ∧
      Not (equivalent left right)

/-!
精确相等版本只适用于已经 canonicalize 的 Graph 协议。匿名节点图或 Bond 列表
未规范化时，必须使用上面的 up-to-equivalence 定义。
-/
def HiddenStrictSample {Sample Graph : Type}
    (candidates : Sample -> CandidateSet Graph) (sample : Sample) : Prop :=
  HiddenStrictSampleUpTo (fun left right : Graph => left = right) candidates sample

def HiddenAmbiguousSample {Sample Graph : Type}
    (candidates : Sample -> CandidateSet Graph) (sample : Sample) : Prop :=
  HiddenAmbiguousSampleUpTo
    (fun left right : Graph => left = right) candidates sample

/-! 每个 hidden pack 都必须非空，并实际含有 strict 与 ambiguity witness。 -/
structure HiddenPackAdequate {Sample Graph : Type}
    (candidates : Sample -> CandidateSet Graph)
    (pack : List Sample) : Prop where
  nodup : pack.Nodup
  nonempty : pack ≠ []
  strictWitness : Exists fun sample =>
    sample ∈ pack ∧ HiddenStrictSample candidates sample
  ambiguityWitness : Exists fun sample =>
    sample ∈ pack ∧ HiddenAmbiguousSample candidates sample

/-!
隔离按显式 `domainKey` 而非样本 ID 判断，防止同一内容换 ID 后跨 A/B。
Lean 不证明该 key 真的是 raster、scaffold 或生成参数；其实现仍需外部审计。
-/
def PacksSeparatedByKey {Sample Key : Type}
    (domainKey : Sample -> Key) (packA packB : List Sample) : Prop :=
  forall sampleA, sampleA ∈ packA ->
    forall sampleB, sampleB ∈ packB -> domainKey sampleA ≠ domainKey sampleB

structure HiddenABDomainSeparated {Sample Graph Key : Type}
    (candidates : Sample -> CandidateSet Graph)
    (domainKey : Sample -> Key)
    (packA packB : List Sample) : Prop where
  packAAdequate : HiddenPackAdequate candidates packA
  packBAdequate : HiddenPackAdequate candidates packB
  domainsSeparated : PacksSeparatedByKey domainKey packA packB

theorem separated_hidden_packs_have_distinct_samples
    {Sample Graph Key : Type}
    (candidates : Sample -> CandidateSet Graph)
    (domainKey : Sample -> Key)
    (packA packB : List Sample)
    (separated : HiddenABDomainSeparated candidates domainKey packA packB) :
    Exists fun sampleA => Exists fun sampleB =>
      sampleA ∈ packA ∧ sampleB ∈ packB ∧ sampleA ≠ sampleB := by
  rcases separated.packAAdequate.strictWitness with
    ⟨sampleA, sampleAIn, _sampleAStrict⟩
  rcases separated.packBAdequate.strictWitness with
    ⟨sampleB, sampleBIn, _sampleBStrict⟩
  refine ⟨sampleA, sampleB, sampleAIn, sampleBIn, ?_⟩
  intro sameSample
  have sameKey : domainKey sampleA = domainKey sampleB := congrArg domainKey sameSample
  exact separated.domainsSeparated sampleA sampleAIn sampleB sampleBIn sameKey

/-!
Prediction commitment 同样只把预测值与 reveal 前发布的 commitment 绑定。`before`
和 `consistentWithReveal` 都是外部协议事实，Lean 不证明网络发布时间或 reveal 保密性。
-/
structure PredictionCommitmentBeforeReveal
    {Prediction Commitment Reveal : Type}
    (commitPrediction : Prediction -> Commitment)
    (before : Commitment -> Reveal -> Prop)
    (consistentWithReveal : Prediction -> Reveal -> Prop)
    (prediction : Prediction) (publishedCommitment : Commitment)
    (reveal : Reveal) : Prop where
  predictionBound : CommitmentBinds
    commitPrediction prediction publishedCommitment
  commitmentUnique : CommitmentUniquelyBinds commitPrediction
  commitmentBeforeReveal : before publishedCommitment reveal
  predictionConsistent : consistentWithReveal prediction reveal

theorem committed_prediction_cannot_be_substituted_after_reveal
    {Prediction Commitment Reveal : Type}
    (commitPrediction : Prediction -> Commitment)
    (before : Commitment -> Reveal -> Prop)
    (consistentWithReveal : Prediction -> Reveal -> Prop)
    (prediction alternative : Prediction)
    (publishedCommitment : Commitment) (reveal : Reveal)
    (protocol : PredictionCommitmentBeforeReveal
      commitPrediction before consistentWithReveal
      prediction publishedCommitment reveal)
    (alternativeBound : CommitmentBinds
      commitPrediction alternative publishedCommitment) :
    alternative = prediction := by
  apply protocol.commitmentUnique
  exact alternativeBound.trans protocol.predictionBound.symm

/-!
finite-complete bucket 用无重复 List 给出声明域的完整有限枚举。非空和异图 pair
是独立字段，防止稳定条件在空域或单图域上真空通过。
-/
def FiniteCompleteDomain {Scene : Type}
    (valid : Scene -> Prop) (scenes : List Scene) : Prop :=
  scenes.Nodup ∧ forall scene, valid scene ↔ scene ∈ scenes

def FinitePairwiseInterGraphSeparationAtLeast
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (scenes : List Scene) (separationLowerBound : Nat) : Prop :=
  forall left, left ∈ scenes -> forall right, right ∈ scenes ->
    Not (equivalent (meaning left) (meaning right)) ->
      separationLowerBound <= metric.dist (render left) (render right)

theorem finite_complete_pairwise_separation_implies_global_separation
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (scenes : List Scene) (separationLowerBound : Nat)
    (complete : FiniteCompleteDomain valid scenes)
    (pairwise : FinitePairwiseInterGraphSeparationAtLeast
      metric equivalent render meaning scenes separationLowerBound) :
    InterGraphSeparationAtLeast metric valid equivalent render meaning
      separationLowerBound := by
  intro left right leftValid rightValid different
  exact pairwise left (complete.2 left |>.mp leftValid)
    right (complete.2 right |>.mp rightValid) different

structure FiniteCompleteStabilityBucket
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (noiseRadius separationLowerBound : Nat)
    (scenes : List Scene) : Prop where
  finiteComplete : FiniteCompleteDomain valid scenes
  nonempty : scenes ≠ []
  inequivalentPair : Exists fun left => Exists fun right =>
    left ∈ scenes ∧ right ∈ scenes ∧
      Not (equivalent (meaning left) (meaning right))
  twiceNoiseBelowSeparation : noiseRadius + noiseRadius < separationLowerBound
  pairwiseSeparation : FinitePairwiseInterGraphSeparationAtLeast
    metric equivalent render meaning scenes separationLowerBound

theorem finite_complete_bucket_covers_every_valid_scene
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (noiseRadius separationLowerBound : Nat) (scenes : List Scene)
    (bucket : FiniteCompleteStabilityBucket metric valid equivalent render meaning
      noiseRadius separationLowerBound scenes) :
    forall scene, valid scene -> scene ∈ scenes := by
  intro scene sceneValid
  exact bucket.finiteComplete.2 scene |>.mp sceneValid

theorem finite_complete_bucket_implies_stability_gate
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (noiseRadius separationLowerBound : Nat) (scenes : List Scene)
    (bucket : FiniteCompleteStabilityBucket metric valid equivalent render meaning
      noiseRadius separationLowerBound scenes) :
    EpsilonDeltaStabilityGate metric valid equivalent render meaning
      noiseRadius separationLowerBound := {
  twiceNoiseBelowSeparation := bucket.twiceNoiseBelowSeparation
  interGraphSeparation :=
    finite_complete_pairwise_separation_implies_global_separation
      metric valid equivalent render meaning scenes separationLowerBound
      bucket.finiteComplete bucket.pairwiseSeparation
}

theorem finite_complete_bucket_implies_auditable_stability
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (noiseRadius separationLowerBound : Nat) (scenes : List Scene)
    (bucket : FiniteCompleteStabilityBucket metric valid equivalent render meaning
      noiseRadius separationLowerBound scenes) :
    AuditableEpsilonDeltaStabilityGate metric valid equivalent render meaning
      noiseRadius separationLowerBound := by
  have sceneExists : Exists fun scene => scene ∈ scenes := by
    cases scenes with
    | nil => exact False.elim (bucket.nonempty rfl)
    | cons head tail => exact ⟨head, by simp⟩
  rcases sceneExists with ⟨scene, sceneIn⟩
  have sceneValid : valid scene :=
    (bucket.finiteComplete.2 scene).mpr sceneIn
  rcases bucket.inequivalentPair with
    ⟨left, right, leftIn, rightIn, different⟩
  exact {
    validDomainNonempty := ⟨scene, sceneValid⟩
    inequivalentPairExists := ⟨left, right,
      (bucket.finiteComplete.2 left).mpr leftIn,
      (bucket.finiteComplete.2 right).mpr rightIn,
      different⟩
    stability := finite_complete_bucket_implies_stability_gate
      metric valid equivalent render meaning noiseRadius separationLowerBound
      scenes bucket
  }

end ChemGraph
