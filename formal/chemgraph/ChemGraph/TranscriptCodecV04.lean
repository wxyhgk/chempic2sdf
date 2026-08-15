import Std

namespace ChemGraph.TranscriptCodecV04

/-! Byte-exact model of the frozen Python v0.4 relation transcript codec. -/

abbrev Bytes := List UInt8

def ascii (value : String) : Bytes := value.toUTF8.data.toList

def beValue (bytes : Bytes) : Nat :=
  bytes.foldl (fun value byte => value * 256 + byte.toNat) 0

def u32be (value : Nat) : Bytes :=
  [UInt8.ofNat (value / 16777216), UInt8.ofNat (value / 65536),
   UInt8.ofNat (value / 256), UInt8.ofNat value]

structure Frame (width : Nat) where
  lengthBytes : Bytes
  lengthWidth : lengthBytes.length = width
  payload : Bytes
  lengthMatches : beValue lengthBytes = payload.length
deriving DecidableEq, Repr

abbrev Frame32 := Frame 4
abbrev Frame64 := Frame 8

structure NonemptyFrame64 where
  frame : Frame64
  nonempty : frame.payload ≠ []
deriving DecidableEq, Repr

inductive Version where | v1
deriving DecidableEq, Repr

inductive RelationId where
  | r01 | r02 | r03 | r04 | r05 | r06 | r07 | r08 | r09 | r10
  | r11 | r12 | r13 | r14 | r15 | r16 | r17 | r18 | r19 | r20
  | r21 | r22 | r23 | r24 | r25 | r26 | r27 | r28 | r29 | r30
  | r31 | r32 | r33 | r34 | r35 | r36 | r37 | r38 | r39 | r40
  | r41 | r42 | r43 | r44 | r45 | r46
deriving DecidableEq, Repr

def allRelations : List RelationId :=
  [.r01, .r02, .r03, .r04, .r05, .r06, .r07, .r08, .r09, .r10,
   .r11, .r12, .r13, .r14, .r15, .r16, .r17, .r18, .r19, .r20,
   .r21, .r22, .r23, .r24, .r25, .r26, .r27, .r28, .r29, .r30,
   .r31, .r32, .r33, .r34, .r35, .r36, .r37, .r38, .r39, .r40,
   .r41, .r42, .r43, .r44, .r45, .r46]

def RelationId.code : RelationId -> Nat
  | .r01 => 1 | .r02 => 2 | .r03 => 3 | .r04 => 4 | .r05 => 5
  | .r06 => 6 | .r07 => 7 | .r08 => 8 | .r09 => 9 | .r10 => 10
  | .r11 => 11 | .r12 => 12 | .r13 => 13 | .r14 => 14 | .r15 => 15
  | .r16 => 16 | .r17 => 17 | .r18 => 18 | .r19 => 19 | .r20 => 20
  | .r21 => 21 | .r22 => 22 | .r23 => 23 | .r24 => 24 | .r25 => 25
  | .r26 => 26 | .r27 => 27 | .r28 => 28 | .r29 => 29 | .r30 => 30
  | .r31 => 31 | .r32 => 32 | .r33 => 33 | .r34 => 34 | .r35 => 35
  | .r36 => 36 | .r37 => 37 | .r38 => 38 | .r39 => 39 | .r40 => 40
  | .r41 => 41 | .r42 => 42 | .r43 => 43 | .r44 => 44 | .r45 => 45
  | .r46 => 46

def relationTag (id : RelationId) : Bytes :=
  let code := id.code
  [0x72, UInt8.ofNat (0x30 + code / 10), UInt8.ofNat (0x30 + code % 10)]

def relationFromTag (tag : Bytes) : Option RelationId :=
  allRelations.find? (fun id => relationTag id == tag)

theorem relationFromTag_relationTag (id : RelationId) :
    relationFromTag (relationTag id) = some id := by
  cases id <;> decide

def expectedOwner : RelationId -> Bytes
  | .r07 | .r30 | .r31 | .r34 | .r35 | .r36 =>
      ascii "outer_deadline_executor+bootstrap_watchdog"
  | .r12 | .r13 | .r14 | .r15 | .r21 | .r42 =>
      ascii "monitor_verified_by_bootstrap_watchdog"
  | .r16 | .r17 | .r18 | .r19 | .r20 | .r32 | .r33 | .r40 | .r41 =>
      ascii "monitor+bootstrap_watchdog"
  | _ => ascii "bootstrap_watchdog"

def relationVersion (id : RelationId) : Bytes :=
  if id = .r29 then ascii "v04" else ascii "v03"

def expectedVerifier (id : RelationId) : Bytes :=
  ascii "chemgraph_e2_" ++ relationVersion id ++ ascii "_verify_" ++ relationTag id

def expectedTheorem (id : RelationId) : Bytes :=
  ascii "chemgraph_e2_" ++ relationVersion id ++ ascii "_relation_" ++
    relationTag id ++ ascii "_sound"

def sourceHash : Bytes :=
  ascii "729f42feb9b5d13833e0fe6f9da1c51d70f55933045f34eee601eb720753cc0a"

def treeHash : Bytes :=
  ascii "3c5307d766fc7376659b22f74f354a912b71ff7c9cabcf90df87c79aec44b7e7"

def domainBytes : Bytes := ascii "chemgraph-e2-v0.4/relation-transcript/v1"

def canonicalFrame32 (payload : Bytes) (fits : payload.length < 16777216) : Frame32 where
  lengthBytes := u32be payload.length
  lengthWidth := by simp [u32be]
  payload := payload
  lengthMatches := by
    simp [u32be, beValue]
    omega

def domainFrame : Frame32 := canonicalFrame32 domainBytes (by decide)
def relationFrame (id : RelationId) : Frame32 :=
  canonicalFrame32 (relationTag id) (by cases id <;> decide)
def ownerFrame (id : RelationId) : Frame32 :=
  canonicalFrame32 (expectedOwner id) (by cases id <;> decide)
def verifierFrame (id : RelationId) : Frame32 :=
  canonicalFrame32 (expectedVerifier id) (by cases id <;> decide)
def theoremFrame (id : RelationId) : Frame32 :=
  canonicalFrame32 (expectedTheorem id) (by cases id <;> decide)
def sourceHashFrame : Frame32 := canonicalFrame32 sourceHash (by decide)
def treeHashFrame : Frame32 := canonicalFrame32 treeHash (by decide)

structure Header where
  version : Version
  relation : RelationId
deriving DecidableEq, Repr

structure InputFrames where
  countBytes : Bytes
  countWidth : countBytes.length = 8
  values : List NonemptyFrame64
  countMatches : beValue countBytes = values.length
  withinFrozenLimit : values.length ≤ 1000000
  nonempty : values ≠ []
deriving DecidableEq, Repr

structure Body where
  orderedInputs : InputFrames
  result : NonemptyFrame64
deriving DecidableEq, Repr

structure Transcript where
  header : Header
  body : Body
deriving DecidableEq, Repr

def Transcript.owner (value : Transcript) : ByteArray :=
  (expectedOwner value.header.relation).toByteArray
def Transcript.verifier (value : Transcript) : ByteArray :=
  (expectedVerifier value.header.relation).toByteArray
def Transcript.theoremId (value : Transcript) : ByteArray :=
  (expectedTheorem value.header.relation).toByteArray

inductive DecodeError where
  | truncated | wrongDomain | unknownRelation | metadataMismatch
  | lengthOverflow | emptyInputs | emptyPayload | trailingBytes | noncanonical
deriving DecidableEq, Repr

def encodeFrame {width : Nat} (frame : Frame width) : Bytes :=
  frame.lengthBytes ++ frame.payload

def encodeHeader (header : Header) : Bytes :=
  encodeFrame domainFrame ++ encodeFrame (relationFrame header.relation) ++
    encodeFrame (ownerFrame header.relation) ++ encodeFrame (verifierFrame header.relation) ++
    encodeFrame (theoremFrame header.relation) ++ encodeFrame sourceHashFrame ++
    encodeFrame treeHashFrame

def encodeInputs (inputs : List NonemptyFrame64) : Bytes :=
  inputs.flatMap (fun input => encodeFrame input.frame)

def encodeBody (body : Body) : Bytes :=
  body.orderedInputs.countBytes ++ encodeInputs body.orderedInputs.values ++
    encodeFrame body.result.frame

def encodeList (value : Transcript) : Bytes := encodeHeader value.header ++ encodeBody value.body
def encode (value : Transcript) : ByteArray := (encodeList value).toByteArray

abbrev Parser (alpha : Type) := Bytes -> Except DecodeError (alpha × Bytes)

def readExact (count : Nat) : Parser Bytes := fun input =>
  if count ≤ input.length then .ok (input.take count, input.drop count)
  else .error .truncated

def expect (expected : Bytes) (error : DecodeError) : Parser Unit := fun input =>
  if input.take expected.length = expected then .ok ((), input.drop expected.length)
  else .error error

@[simp] theorem readExact_append {count : Nat} {head tail : Bytes}
    (size : head.length = count) :
    readExact count (head ++ tail) = .ok (head, tail) := by simp [readExact, size]

@[simp] theorem readExact_append_append {count : Nat} {head middle tail : Bytes}
    (size : head.length = count) :
    readExact count (head ++ middle ++ tail) = .ok (head, middle ++ tail) := by
  rw [List.append_assoc]
  exact readExact_append size

@[simp] theorem expect_append (expected rest : Bytes) (error : DecodeError) :
    expect expected error (expected ++ rest) = .ok ((), rest) := by simp [expect]

@[simp] theorem except_ok_bind {epsilon alpha beta : Type}
    (value : alpha) (next : alpha -> Except epsilon beta) :
    (Except.ok value >>= next) = next value := rfl

def parseFrame (width : Nat) : Parser (Frame width) := fun input => do
  let (lengthBytes, rest) <- readExact width input
  if widthProof : lengthBytes.length = width then
    let length := beValue lengthBytes
    if length > rest.length then .error .truncated
    else
      let (payload, tail) <- readExact length rest
      if lengthProof : length = payload.length then
        .ok (⟨lengthBytes, widthProof, payload, lengthProof⟩, tail)
      else .error .noncanonical
  else .error .noncanonical

@[simp] theorem parseFrame_encode {width : Nat} (frame : Frame width) (rest : Bytes) :
    parseFrame width (encodeFrame frame ++ rest) = .ok (frame, rest) := by
  rcases frame with ⟨lengthBytes, lengthWidth, payload, lengthMatches⟩
  unfold parseFrame encodeFrame
  rw [readExact_append_append lengthWidth]
  simp only [except_ok_bind]
  rw [dif_pos lengthWidth]
  have enough : beValue lengthBytes ≤ (payload ++ rest).length := by
    simp [lengthMatches]
  simp only [gt_iff_lt, if_neg (Nat.not_lt.mpr enough)]
  rw [readExact_append lengthMatches.symm]
  simp only [except_ok_bind]
  rw [dif_pos lengthMatches]

def parseRelation : Parser RelationId := fun input => do
  let (frame, rest) <- parseFrame 4 input
  match relationFromTag frame.payload with
  | none => .error .unknownRelation
  | some id =>
      if frame = relationFrame id then .ok (id, rest)
      else .error .noncanonical

@[simp] theorem parseRelation_encode (id : RelationId) (rest : Bytes) :
    parseRelation (encodeFrame (relationFrame id) ++ rest) = .ok (id, rest) := by
  unfold parseRelation
  rw [parseFrame_encode]
  simp only [except_ok_bind]
  change (match relationFromTag (relationTag id) with
    | none => Except.error DecodeError.unknownRelation
    | some decoded => if relationFrame id = relationFrame decoded then
        Except.ok (decoded, rest) else Except.error DecodeError.noncanonical) =
          Except.ok (id, rest)
  rw [relationFromTag_relationTag]
  simp

def parseHeader : Parser Header := fun input => do
  let (_, s1) <- expect (encodeFrame domainFrame) .wrongDomain input
  let (relation, s2) <- parseRelation s1
  let (_, s3) <- expect (encodeFrame (ownerFrame relation)) .metadataMismatch s2
  let (_, s4) <- expect (encodeFrame (verifierFrame relation)) .metadataMismatch s3
  let (_, s5) <- expect (encodeFrame (theoremFrame relation)) .metadataMismatch s4
  let (_, s6) <- expect (encodeFrame sourceHashFrame) .metadataMismatch s5
  let (_, rest) <- expect (encodeFrame treeHashFrame) .metadataMismatch s6
  .ok (⟨.v1, relation⟩, rest)

@[simp] theorem parseHeader_encode (header : Header) (rest : Bytes) :
    parseHeader (encodeHeader header ++ rest) = .ok (header, rest) := by
  rcases header with ⟨version, relation⟩
  cases version
  unfold parseHeader encodeHeader
  simp only [List.append_assoc, expect_append, except_ok_bind, parseRelation_encode]

def parseNonemptyFrame64 : Parser NonemptyFrame64 := fun input => do
  let (frame, rest) <- parseFrame 8 input
  if nonempty : frame.payload ≠ [] then .ok (⟨frame, nonempty⟩, rest)
  else .error .emptyPayload

@[simp] theorem parseNonemptyFrame64_encode (frame : NonemptyFrame64) (rest : Bytes) :
    parseNonemptyFrame64 (encodeFrame frame.frame ++ rest) = .ok (frame, rest) := by
  rcases frame with ⟨frame, nonempty⟩
  unfold parseNonemptyFrame64
  rw [parseFrame_encode]
  simp only [except_ok_bind]
  rw [dif_pos nonempty]

def parseInputs : Nat -> Parser (List NonemptyFrame64)
  | 0 => fun input => .ok ([], input)
  | count + 1 => fun input => do
      let (head, s1) <- parseNonemptyFrame64 input
      let (tail, rest) <- parseInputs count s1
      .ok (head :: tail, rest)

@[simp] theorem parseInputs_encode (inputs : List NonemptyFrame64) (rest : Bytes) :
    parseInputs inputs.length (encodeInputs inputs ++ rest) = .ok (inputs, rest) := by
  induction inputs with
  | nil => simp [parseInputs, encodeInputs]
  | cons head tail inductionHypothesis =>
      simp only [List.length_cons, parseInputs, encodeInputs, List.flatMap_cons,
        List.append_assoc, parseNonemptyFrame64_encode, except_ok_bind]
      change (do
        let (decodedTail, decodedRest) <- parseInputs tail.length (encodeInputs tail ++ rest)
        .ok (head :: decodedTail, decodedRest)) = .ok (head :: tail, rest)
      rw [inductionHypothesis]
      simp only [except_ok_bind]

def parseInputBlock : Parser InputFrames := fun input => do
  let (countBytes, rest) <- readExact 8 input
  if countWidth : countBytes.length = 8 then
    let count := beValue countBytes
    if zero : count = 0 then .error .emptyInputs
    else if overflow : count > 1000000 then .error .lengthOverflow
    else
      let (values, tail) <- parseInputs count rest
      if countMatches : count = values.length then
        .ok (⟨countBytes, countWidth, values, countMatches,
          by
            rw [← countMatches]
            exact Nat.le_of_not_gt overflow,
          by
            intro empty
            apply zero
            simpa [empty] using countMatches⟩, tail)
      else .error .noncanonical
  else .error .noncanonical

@[simp] theorem parseInputBlock_encode (inputs : InputFrames) (rest : Bytes) :
    parseInputBlock (inputs.countBytes ++ encodeInputs inputs.values ++ rest) =
      .ok (inputs, rest) := by
  rcases inputs with ⟨countBytes, countWidth, values, countMatches, limit, nonempty⟩
  unfold parseInputBlock
  rw [readExact_append_append countWidth]
  simp only [except_ok_bind]
  rw [dif_pos countWidth]
  have positive : beValue countBytes ≠ 0 := by
    intro zero
    apply nonempty
    apply List.eq_nil_of_length_eq_zero
    rw [← countMatches]
    exact zero
  rw [dif_neg positive]
  have notOverflow : ¬ beValue countBytes > 1000000 := by
    rw [countMatches]
    omega
  rw [dif_neg notOverflow]
  have parsed : parseInputs (beValue countBytes) (encodeInputs values ++ rest) =
      .ok (values, rest) := by
    rw [countMatches]
    exact parseInputs_encode values rest
  rw [parsed]
  simp only [except_ok_bind]
  rw [dif_pos countMatches]

def parseBody : Parser Body := fun input => do
  let (inputs, rest) <- parseInputBlock input
  let (result, tail) <- parseNonemptyFrame64 rest
  .ok (⟨inputs, result⟩, tail)

@[simp] theorem parseBody_encode (body : Body) (rest : Bytes) :
    parseBody (encodeBody body ++ rest) = .ok (body, rest) := by
  rcases body with ⟨inputs, result⟩
  unfold parseBody encodeBody
  rw [List.append_assoc (inputs.countBytes ++ encodeInputs inputs.values)
    (encodeFrame result.frame) rest]
  rw [parseInputBlock_encode inputs (encodeFrame result.frame ++ rest)]
  simp only [except_ok_bind]
  rw [parseNonemptyFrame64_encode]
  simp only [except_ok_bind]

def parseList : Parser Transcript := fun input => do
  let (header, rest) <- parseHeader input
  let (body, tail) <- parseBody rest
  .ok (⟨header, body⟩, tail)

@[simp] theorem parseList_encode (value : Transcript) (rest : Bytes) :
    parseList (encodeList value ++ rest) = .ok (value, rest) := by
  rcases value with ⟨header, body⟩
  unfold parseList encodeList
  rw [List.append_assoc, parseHeader_encode]
  simp only [except_ok_bind]
  rw [parseBody_encode]
  simp only [except_ok_bind]

def decodeList (bytes : Bytes) : Except DecodeError Transcript :=
  match parseList bytes with
  | .error error => .error error
  | .ok (value, rest) =>
      if rest ≠ [] then .error .trailingBytes
      else if encodeList value = bytes then .ok value
      else .error .noncanonical

def decode (bytes : ByteArray) : Except DecodeError Transcript := decodeList bytes.data.toList

theorem decodeList_encode (value : Transcript) : decodeList (encodeList value) = .ok value := by
  unfold decodeList
  have parsed : parseList (encodeList value) = .ok (value, []) := by
    simpa using parseList_encode value []
  rw [parsed]
  simp

theorem decodeList_success_reencode {bytes : Bytes} {value : Transcript}
    (success : decodeList bytes = .ok value) : encodeList value = bytes := by
  unfold decodeList at success
  cases parsedEq : parseList bytes with
  | error error => simp [parsedEq] at success
  | ok parsed =>
    rw [parsedEq] at success
    rcases parsed with ⟨decoded, rest⟩
    simp only at success
    by_cases hasTrailing : rest ≠ []
    · rw [if_pos hasTrailing] at success
      contradiction
    · rw [if_neg hasTrailing] at success
      by_cases canonical : encodeList decoded = bytes
      · rw [if_pos canonical] at success
        have sameValue : decoded = value := Except.ok.inj success
        simpa [sameValue] using canonical
      · rw [if_neg canonical] at success
        contradiction

theorem decodeList_rejects_trailing (value : Transcript) {trailing : Bytes}
    (nonempty : trailing ≠ []) :
    decodeList (encodeList value ++ trailing) = .error .trailingBytes := by
  unfold decodeList
  rw [parseList_encode]
  simp [nonempty]

@[simp] theorem encode_data_toList (value : Transcript) :
    (encode value).data.toList = encodeList value := by simp [encode]

theorem decode_encode (value : Transcript) : decode (encode value) = .ok value := by
  simp [decode, decodeList_encode]

theorem decode_success_reencode {bytes : ByteArray} {value : Transcript}
    (success : decode bytes = .ok value) : encode value = bytes := by
  have exactList := decodeList_success_reencode (bytes := bytes.data.toList) success
  apply ByteArray.ext
  apply Array.ext'
  simpa [encode] using exactList

theorem encode_injective : Function.Injective encode := by
  intro left right same
  have leftDecoded : decode (encode left) = .ok left := decode_encode left
  have rightDecoded : decode (encode right) = .ok right := decode_encode right
  rw [same, rightDecoded] at leftDecoded
  exact (Except.ok.inj leftDecoded).symm

def hexNibble (byte : UInt8) : Option Nat :=
  if 0x30 ≤ byte ∧ byte ≤ 0x39 then some (byte.toNat - 0x30)
  else if 0x61 ≤ byte ∧ byte ≤ 0x66 then some (byte.toNat - 0x61 + 10)
  else none

def decodeHex : Bytes -> Option Bytes
  | [] => some []
  | high :: low :: rest => do
      let highValue <- hexNibble high
      let lowValue <- hexNibble low
      let tail <- decodeHex rest
      some (UInt8.ofNat (highValue * 16 + lowValue) :: tail)
  | [_] => none

def r01FixtureHex : String :=
  "000000286368656d67726170682d65322d76302e342f72656c6174696f6e2d7472616e7363726970742f76310000000372303100000012626f6f7473747261705f7761746368646f670000001b6368656d67726170685f65325f7630335f7665726966795f723031000000236368656d67726170685f65325f7630335f72656c6174696f6e5f7230315f736f756e6400000040373239663432666562396235643133383333653066653666396461316335316437306635353933333034356633346565653630316562373230373533636330610000004033633533303764373636666337333736363539623232663734663335346139313262373166663763396361626366393064663837633739616563343462376537000000000000000200000000000000037b7d0a00000000000000147b22737461747573223a22706173736564227d0a000000000000001201636f6e74726163742d62696e64696e6773"

def r01FixtureBytes : Bytes := (decodeHex (ascii r01FixtureHex)).getD []

def r01Input1 : NonemptyFrame64 where
  frame := {
    lengthBytes := [0, 0, 0, 0, 0, 0, 0, 3]
    lengthWidth := by decide
    payload := [0x7b, 0x7d, 0x0a]
    lengthMatches := by decide
  }
  nonempty := by decide

def r01Input2 : NonemptyFrame64 where
  frame := {
    lengthBytes := [0, 0, 0, 0, 0, 0, 0, 20]
    lengthWidth := by decide
    payload := ascii "{\"status\":\"passed\"}\n"
    lengthMatches := by decide
  }
  nonempty := by decide

def r01Result : NonemptyFrame64 where
  frame := {
    lengthBytes := [0, 0, 0, 0, 0, 0, 0, 18]
    lengthWidth := by decide
    payload := 0x01 :: ascii "contract-bindings"
    lengthMatches := by decide
  }
  nonempty := by decide

def r01FixtureTranscript : Transcript where
  header := ⟨.v1, .r01⟩
  body := {
    orderedInputs := {
      countBytes := [0, 0, 0, 0, 0, 0, 0, 2]
      countWidth := by decide
      values := [r01Input1, r01Input2]
      countMatches := by decide
      withinFrozenLimit := by decide
      nonempty := by decide
    }
    result := r01Result
  }

set_option maxRecDepth 4096 in
theorem r01_fixture_hex_decodes : decodeHex (ascii r01FixtureHex) = some r01FixtureBytes := by
  decide

set_option maxRecDepth 4096 in
theorem r01_fixture_byte_length : r01FixtureBytes.length = 352 := by decide

set_option maxRecDepth 4096 in
theorem r01_fixture_exact_bytes :
    encodeList r01FixtureTranscript = r01FixtureBytes := by decide

theorem r01_fixture_decodes :
    decode r01FixtureBytes.toByteArray = .ok r01FixtureTranscript := by
  have encoded := decode_encode r01FixtureTranscript
  rw [← r01_fixture_exact_bytes]
  simpa [decode, encode, List.toList_data_toByteArray] using encoded

theorem decode_rejects_trailing_bytes (value : Transcript) {trailing : Bytes}
    (nonempty : trailing ≠ []) :
    decode (encodeList value ++ trailing).toByteArray = .error .trailingBytes := by
  simpa [decode, List.toList_data_toByteArray] using
    decodeList_rejects_trailing value nonempty

end ChemGraph.TranscriptCodecV04
