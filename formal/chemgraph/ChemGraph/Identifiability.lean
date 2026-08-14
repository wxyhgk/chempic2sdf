namespace ChemGraph

/-! 渲染在场景集合上可辨识，当且仅当同一观测不会对应不同语义图。 -/
def Identifiable {Scene Observation Graph : Type}
    (render : Scene -> Observation) (meaning : Scene -> Graph) : Prop :=
  forall s t, render s = render t -> meaning s = meaning t

def IdentifiableOn {Scene Observation Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph) : Prop :=
  forall s t, valid s -> valid t -> render s = render t -> meaning s = meaning t

def IdentifiableUpTo {Scene Observation Graph : Type}
    (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Observation) (meaning : Scene -> Graph) : Prop :=
  forall s t, render s = render t -> equivalent (meaning s) (meaning t)

def IdentifiableOnUpTo {Scene Observation Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Observation) (meaning : Scene -> Graph) : Prop :=
  forall s t, valid s -> valid t -> render s = render t ->
    equivalent (meaning s) (meaning t)

def PriorCoversBaseObservations {Scene Observation : Type}
    (base prior : Scene -> Prop) (render : Scene -> Observation) : Prop :=
  forall source, base source -> Exists fun kept =>
    And (base kept) (And (prior kept) (render kept = render source))

/-! 一个先验只有在缩小允许场景集合后恢复了可辨识性，才叫真正解决歧义。 -/
def ResolvedByPrior {Scene Observation Graph : Type}
    (base prior : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph) : Prop :=
  And (Not (IdentifiableOn base render meaning))
    (And (PriorCoversBaseObservations base prior render)
      (IdentifiableOn (fun scene => And (base scene) (prior scene)) render meaning))

def AmbiguousObservation {Scene Observation Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Observation)
    (meaning : Scene -> Graph) (observation : Observation) : Prop :=
  Exists fun s => Exists fun t =>
    And (valid s) (And (valid t)
      (And (render s = observation)
        (And (render t = observation) (Not (equivalent (meaning s) (meaning t))))))

theorem collision_forbids_decoder_up_to
    {Scene Observation Graph : Type}
    (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (s t : Scene)
    (sameObservation : render s = render t)
    (differentMeaning : Not (equivalent (meaning s) (meaning t))) :
    Not (Exists fun decode : Observation -> Graph =>
      forall scene, equivalent (decode (render scene)) (meaning scene)) := by
  intro existsDecoder
  rcases existsDecoder with ⟨decode, exactUpTo⟩
  apply differentMeaning
  exact equivalenceLaw.trans
    (equivalenceLaw.symm (exactUpTo s))
    (sameObservation ▸ exactUpTo t)

theorem leftInverse_implies_identifiable
    {Scene Observation Graph : Type}
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (decode : Observation -> Graph)
    (roundtrip : forall scene, decode (render scene) = meaning scene) :
    Identifiable render meaning := by
  intro s t sameObservation
  calc
    meaning s = decode (render s) := (roundtrip s).symm
    _ = decode (render t) := congrArg decode sameObservation
    _ = meaning t := roundtrip t

/-!
如果两个不同语义图产生完全相同的观测，则不存在对全部场景都正确的
确定性解码器。这一结论与模型大小、训练量和优化算法无关。
-/
theorem collision_forbids_exact_decoder
    {Scene Observation Graph : Type}
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (s t : Scene)
    (sameObservation : render s = render t)
    (differentMeaning : meaning s ≠ meaning t) :
    Not (Exists fun decode : Observation -> Graph =>
      forall scene, decode (render scene) = meaning scene) := by
  intro existsDecoder
  rcases existsDecoder with ⟨decode, exactDecode⟩
  apply differentMeaning
  calc
    meaning s = decode (render s) := (exactDecode s).symm
    _ = decode (render t) := congrArg decode sameObservation
    _ = meaning t := exactDecode t

/-! 粗观测发生碰撞而丰富观测不同，则丰富观测不可能是粗观测的确定性函数。 -/
theorem collision_forbids_factorization
    {Scene Coarse Rich : Type}
    (coarse : Scene -> Coarse) (rich : Scene -> Rich)
    (s t : Scene)
    (sameCoarse : coarse s = coarse t)
    (differentRich : rich s ≠ rich t) :
    Not (Exists fun lift : Coarse -> Rich =>
      forall scene, lift (coarse scene) = rich scene) := by
  intro existsLift
  rcases existsLift with ⟨lift, factors⟩
  apply differentRich
  calc
    rich s = lift (coarse s) := (factors s).symm
    _ = lift (coarse t) := congrArg lift sameCoarse
    _ = rich t := factors t

end ChemGraph
