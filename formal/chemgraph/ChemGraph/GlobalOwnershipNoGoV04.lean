import Std

namespace ChemGraph

namespace GlobalOwnershipNoGoV04

/-!
Global Ownership v0.4 的 observation-collision NO-GO。

OwnerPartition 用 total function 表达每个 Pixel 的唯一 Owner。World 另外保存
intersection support；`legal` 由调用者显式提供。若两份 legal World 具有同一
Observation、但要求不同 owner partition，则只读 Observation 的 deterministic
constructor 不可能对两者都 exact。若 support 不同，同样不能 exact 构造同时包含
support 与 partition 的 OwnershipInterpretation。

本模块不构造真实 PNG witness，不定义 projection/support extractor，也不证明调用者
提供的 `legal` predicate 与任何真实图像一致。
-/

structure Observation (RootEvidence : Type) where
  rootEvidence : RootEvidence

/-- total function 使每个 Pixel 按构造只有一个 Owner。 -/
structure OwnerPartition (Pixel Owner : Type) where
  ownerOf : Pixel -> Owner

def OwnerPartition.Owns
    {Pixel Owner : Type}
    (partition : OwnerPartition Pixel Owner)
    (pixel : Pixel) (owner : Owner) : Prop :=
  partition.ownerOf pixel = owner

def OwnerPartition.HasUniqueOwner
    {Pixel Owner : Type}
    (partition : OwnerPartition Pixel Owner)
    (pixel : Pixel) : Prop :=
  Exists fun owner : Owner =>
    partition.Owns pixel owner /\
      forall other : Owner, partition.Owns pixel other -> other = owner

theorem ownerPartition_pixel_has_unique_owner
    {Pixel Owner : Type}
    (partition : OwnerPartition Pixel Owner)
    (pixel : Pixel) :
    partition.HasUniqueOwner pixel := by
  refine ⟨partition.ownerOf pixel, rfl, ?_⟩
  intro other owned
  exact owned.symm

structure OwnershipInterpretation (Pixel Owner : Type) where
  intersectionSupport : Pixel -> Prop
  ownerPartition : OwnerPartition Pixel Owner

structure World (RootEvidence Pixel Owner : Type) where
  observation : Observation RootEvidence
  intersectionSupport : Pixel -> Prop
  ownerPartition : OwnerPartition Pixel Owner

def World.interpretation
    {RootEvidence Pixel Owner : Type}
    (world : World RootEvidence Pixel Owner) :
    OwnershipInterpretation Pixel Owner :=
  { intersectionSupport := world.intersectionSupport
    ownerPartition := world.ownerPartition }

def RootOnlyExactOwnershipConstructor
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (construct : Observation RootEvidence ->
      OwnershipInterpretation Pixel Owner) : Prop :=
  forall world, legal world ->
    construct world.observation = world.interpretation

def RootOnlyExactUniqueOwnerConstructor
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (construct : Observation RootEvidence -> OwnerPartition Pixel Owner) : Prop :=
  forall world, legal world ->
    construct world.observation = world.ownerPartition

theorem same_observation_different_interpretations_forbid_root_only_exact_constructor
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (left right : World RootEvidence Pixel Owner)
    (leftLegal : legal left) (rightLegal : legal right)
    (sameObservation : left.observation = right.observation)
    (differentInterpretation : left.interpretation ≠ right.interpretation) :
    Not (Exists fun construct : Observation RootEvidence ->
      OwnershipInterpretation Pixel Owner =>
      RootOnlyExactOwnershipConstructor legal construct) := by
  rintro ⟨construct, exact⟩
  apply differentInterpretation
  calc
    left.interpretation = construct left.observation :=
      (exact left leftLegal).symm
    _ = construct right.observation := congrArg construct sameObservation
    _ = right.interpretation := exact right rightLegal

/-- partition 不同已经足以排除 root-only exact unique-owner constructor。 -/
theorem same_observation_different_owner_partitions_forbid_unique_owner_constructor
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (left right : World RootEvidence Pixel Owner)
    (leftLegal : legal left) (rightLegal : legal right)
    (sameObservation : left.observation = right.observation)
    (differentPartition : left.ownerPartition ≠ right.ownerPartition) :
    Not (Exists fun construct : Observation RootEvidence ->
      OwnerPartition Pixel Owner =>
      RootOnlyExactUniqueOwnerConstructor legal construct) := by
  rintro ⟨construct, exact⟩
  apply differentPartition
  calc
    left.ownerPartition = construct left.observation :=
      (exact left leftLegal).symm
    _ = construct right.observation := congrArg construct sameObservation
    _ = right.ownerPartition := exact right rightLegal

theorem different_partitions_imply_different_interpretations
    {RootEvidence Pixel Owner : Type}
    {left right : World RootEvidence Pixel Owner}
    (differentPartition : left.ownerPartition ≠ right.ownerPartition) :
    left.interpretation ≠ right.interpretation := by
  intro sameInterpretation
  apply differentPartition
  exact congrArg OwnershipInterpretation.ownerPartition sameInterpretation

theorem different_support_implies_different_interpretations
    {RootEvidence Pixel Owner : Type}
    {left right : World RootEvidence Pixel Owner}
    (differentSupport : left.intersectionSupport ≠ right.intersectionSupport) :
    left.interpretation ≠ right.interpretation := by
  intro sameInterpretation
  apply differentSupport
  exact congrArg OwnershipInterpretation.intersectionSupport sameInterpretation

theorem same_observation_different_support_forbids_exact_ownership_interpretation
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (left right : World RootEvidence Pixel Owner)
    (leftLegal : legal left) (rightLegal : legal right)
    (sameObservation : left.observation = right.observation)
    (differentSupport : left.intersectionSupport ≠ right.intersectionSupport) :
    Not (Exists fun construct : Observation RootEvidence ->
      OwnershipInterpretation Pixel Owner =>
      RootOnlyExactOwnershipConstructor legal construct) :=
  same_observation_different_interpretations_forbid_root_only_exact_constructor
    legal left right leftLegal rightLegal sameObservation
      (different_support_implies_different_interpretations differentSupport)

/-!
下面只给一个有限逻辑 witness，证明 NO-GO 前提可满足；它不是 PNG 或化学数据 witness。
-/

def abstractObservation : Observation Unit where
  rootEvidence := ()

def abstractLeftWorld : World Unit Unit Bool where
  observation := abstractObservation
  intersectionSupport := fun _ => True
  ownerPartition := { ownerOf := fun _ => false }

def abstractRightWorld : World Unit Unit Bool where
  observation := abstractObservation
  intersectionSupport := fun _ => True
  ownerPartition := { ownerOf := fun _ => true }

def abstractLegal (_ : World Unit Unit Bool) : Prop := True

theorem abstract_worlds_have_different_owner_partitions :
    abstractLeftWorld.ownerPartition ≠ abstractRightWorld.ownerPartition := by
  intro equal
  have ownerEqual := congrArg
    (fun partition : OwnerPartition Unit Bool => partition.ownerOf ()) equal
  cases ownerEqual

theorem abstract_collision_forbids_unique_owner_constructor :
    Not (Exists fun construct : Observation Unit -> OwnerPartition Unit Bool =>
      RootOnlyExactUniqueOwnerConstructor abstractLegal construct) :=
  same_observation_different_owner_partitions_forbid_unique_owner_constructor
    abstractLegal abstractLeftWorld abstractRightWorld trivial trivial rfl
      abstract_worlds_have_different_owner_partitions

#print axioms ownerPartition_pixel_has_unique_owner
#print axioms same_observation_different_interpretations_forbid_root_only_exact_constructor
#print axioms same_observation_different_owner_partitions_forbid_unique_owner_constructor
#print axioms different_partitions_imply_different_interpretations
#print axioms different_support_implies_different_interpretations
#print axioms same_observation_different_support_forbids_exact_ownership_interpretation
#print axioms abstract_worlds_have_different_owner_partitions
#print axioms abstract_collision_forbids_unique_owner_constructor

end GlobalOwnershipNoGoV04

end ChemGraph
