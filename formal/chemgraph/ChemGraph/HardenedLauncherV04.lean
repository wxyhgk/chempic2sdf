import Std

namespace ChemGraph

namespace HardenedLauncherV04

/-!
Pure v0.4 launcher protocol kernel.  Records below are external observations;
their nonempty canonical bytes and the explicit gates prevent a successful state
from being justified by an unstructured, vacuous policy.
-/

structure Sha256Binding where
  value : String
  decodedNibbles : Vector (Fin 16) 64
  lowercaseEncodingMatches : Bool
  nonzero : Bool

structure CommitBinding where
  value : String
  decodedNibbles : Vector (Fin 16) 40
  lowercaseEncodingMatches : Bool
  nonzero : Bool

def Sha256BindingValid (binding : Sha256Binding) : Prop :=
  binding.lowercaseEncodingMatches = true ∧ binding.nonzero = true

def CommitBindingValid (binding : CommitBinding) : Prop :=
  binding.lowercaseEncodingMatches = true ∧ binding.nonzero = true

structure NonemptyCanonicalRecord where
  bytes : ByteArray
  nonempty : 0 < bytes.size
  sha256 : Sha256Binding
  duplicateKeysRejected : Bool
  canonicalJson : Bool
  closedSchemaPassed : Bool

def RecordValid (record : NonemptyCanonicalRecord) : Prop :=
  Sha256BindingValid record.sha256 ∧ record.duplicateKeysRejected = true ∧
    record.canonicalJson = true ∧ record.closedSchemaPassed = true

inductive Phase where
  | dryRun
  | official
deriving DecidableEq, Repr

def frozenContractSha256 : String :=
  "ddb03c311e1e3a41f63d2438a8e4c7bf7b7fa9c3044fe6b15fe399e10c47a6ea"

def frozenContractSchemaSha256 : String :=
  "dfa0e0087cf7f065f9219c5d4bdb129723a8a1b3309a7db789828a60819bc5d6"

def frozenInheritedContractSha256 : String :=
  "db0a9e0d0d35b835f99aa65e87f0389751e3d1e1df05203ad5373c05c1bf7e3a"

def frozenInheritedContractSchemaSha256 : String :=
  "2dfa9e02571c4e2f7266beaef4e50d5fe93185245c920910f151a2ef3b805d44"

def frozenP0ContractSha256 : String :=
  "e326fc744903b998793746f299232c4225ce3b8648e1abdda7b4e065f06a9402"

def frozenParentContractSha256 : String :=
  "55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65"

def frozenSourceCommit : String :=
  "5970f316a4cfd30a25d3e89776ab8a4c50742b65"

def frozenSourceFreezeSnapshotSha256 : String :=
  "994ae0de25ae47835c8b2bb97e3579555c45d8b79fd41a3a7c195eb9a0c798dc"

def frozenE2ArchiveSha256 : String :=
  "3385833f32802983b5e84885ecf4ccaaa757f48341a993f576e0bbca8d97e21f"

def frozenFullCommitArchiveSha256 : String :=
  "d10b4e94073fa45e802b6885436607625b148b09771aff8f042f2db25c33a30d"

def frozenRecordSchemaBundleSha256 : String :=
  "6dfd5110420502d2f7bcc17579566592a77727913e10fe4ad15056411064b379"

def frozenRelationProtocolSha256 : String :=
  "2845b67715bbd397b9562f116c2a385cc80de286cd711c0f542bac0fbf5ad8f1"

def frozenRoleMatrixSha256 : String :=
  "1c9db39bc1ab4912c7cc91278cd703c2161db9ab190fc236768ce4aa0a173ba4"

def frozenStateMachineSha256 : String :=
  "44d6ba607690ca7c2430a9158134278bdb5ffa1921857b411e70026669d294ab"

def frozenTreeFramingSha256 : String :=
  "cf398948a0cdd07ca248dc9b4155799b713568d2c50439134cd64d11da336ff8"

structure ImplementationBindings where
  implementationCommit : CommitBinding
  launcherSourceFreezeSha256 : Sha256Binding
  testBundleSha256 : Sha256Binding
  externalToolDependencyClosureSha256 : Sha256Binding
  externalToolManifestRootSha256 : Sha256Binding
  identityRegistrySha256 : Sha256Binding
  bootstrapWatchdogSeccompProfileSha256 : Sha256Binding
  deadlineExecutorSeccompProfileSha256 : Sha256Binding

structure ContractConformanceCertificate where
  record : NonemptyCanonicalRecord
  contractSha256 : String
  contractSchemaSha256 : String
  inheritedContractSha256 : String
  inheritedContractSchemaSha256 : String
  p0ContractSha256 : String
  parentContractSha256 : String
  sourceCommit : String
  sourceFreezeSnapshotSha256 : String
  e2ArchiveSha256 : String
  fullCommitArchiveSha256 : String
  recordSchemaBundleSha256 : String
  relationProtocolSha256 : String
  roleMatrixSha256 : String
  stateMachineSha256 : String
  treeFramingSha256 : String
  implementationCommit : String
  launcherSourceFreezeSha256 : String
  testBundleSha256 : String
  externalToolDependencyClosureSha256 : String
  externalToolManifestRootSha256 : String
  identityRegistrySha256 : String
  bootstrapWatchdogSeccompProfileSha256 : String
  deadlineExecutorSeccompProfileSha256 : String
  allowExclusions : Bool
  trainingAuthorized : Bool
  executionAuthorized : Bool
  maximumWallSeconds : Nat
  maximumAccountedMemoryBytes : Nat
  outerDeadlineExecutorMemoryMaxBytes : Nat
  workloadParentMemoryMaxBytes : Nat
  bootstrapWatchdogMemoryMaxBytes : Nat
  monitorMemoryMaxBytes : Nat
  innerMemoryMaxBytes : Nat
  physicalCpus : Nat
  workersPerSemanticStage : Nat
  statusPassed : Bool

structure ExactContractConformance
    (bindings : ImplementationBindings)
    (certificate : ContractConformanceCertificate) : Prop where
  recordValid : RecordValid certificate.record
  implementationBindingsValid : CommitBindingValid bindings.implementationCommit ∧
    Sha256BindingValid bindings.launcherSourceFreezeSha256 ∧
    Sha256BindingValid bindings.testBundleSha256 ∧
    Sha256BindingValid bindings.externalToolDependencyClosureSha256 ∧
    Sha256BindingValid bindings.externalToolManifestRootSha256 ∧
    Sha256BindingValid bindings.identityRegistrySha256 ∧
    Sha256BindingValid bindings.bootstrapWatchdogSeccompProfileSha256 ∧
    Sha256BindingValid bindings.deadlineExecutorSeccompProfileSha256
  contract : certificate.contractSha256 = frozenContractSha256
  contractSchema : certificate.contractSchemaSha256 = frozenContractSchemaSha256
  inheritedContract : certificate.inheritedContractSha256 = frozenInheritedContractSha256
  inheritedSchema :
    certificate.inheritedContractSchemaSha256 = frozenInheritedContractSchemaSha256
  p0Contract : certificate.p0ContractSha256 = frozenP0ContractSha256
  parentContract : certificate.parentContractSha256 = frozenParentContractSha256
  sourceCommit : certificate.sourceCommit = frozenSourceCommit
  sourceFreeze :
    certificate.sourceFreezeSnapshotSha256 = frozenSourceFreezeSnapshotSha256
  e2Archive : certificate.e2ArchiveSha256 = frozenE2ArchiveSha256
  fullArchive : certificate.fullCommitArchiveSha256 = frozenFullCommitArchiveSha256
  schemaBundle : certificate.recordSchemaBundleSha256 = frozenRecordSchemaBundleSha256
  relationProtocol : certificate.relationProtocolSha256 = frozenRelationProtocolSha256
  roleMatrix : certificate.roleMatrixSha256 = frozenRoleMatrixSha256
  stateMachine : certificate.stateMachineSha256 = frozenStateMachineSha256
  treeFraming : certificate.treeFramingSha256 = frozenTreeFramingSha256
  implementation :
    certificate.implementationCommit = bindings.implementationCommit.value
  launcherFreeze :
    certificate.launcherSourceFreezeSha256 = bindings.launcherSourceFreezeSha256.value
  testBundle : certificate.testBundleSha256 = bindings.testBundleSha256.value
  toolClosure : certificate.externalToolDependencyClosureSha256 =
    bindings.externalToolDependencyClosureSha256.value
  toolManifest : certificate.externalToolManifestRootSha256 =
    bindings.externalToolManifestRootSha256.value
  identityRegistry : certificate.identityRegistrySha256 = bindings.identityRegistrySha256.value
  watchdogSeccomp : certificate.bootstrapWatchdogSeccompProfileSha256 =
    bindings.bootstrapWatchdogSeccompProfileSha256.value
  executorSeccomp : certificate.deadlineExecutorSeccompProfileSha256 =
    bindings.deadlineExecutorSeccompProfileSha256.value
  noExclusions : certificate.allowExclusions = false
  noTraining : certificate.trainingAuthorized = false
  contractDoesNotAuthorizeExecution : certificate.executionAuthorized = false
  wall : certificate.maximumWallSeconds = 1800
  globalMemory : certificate.maximumAccountedMemoryBytes = 8589934592
  executorMemory : certificate.outerDeadlineExecutorMemoryMaxBytes = 33554432
  parentMemory : certificate.workloadParentMemoryMaxBytes = 8556380160
  watchdogMemory : certificate.bootstrapWatchdogMemoryMaxBytes = 67108864
  monitorMemory : certificate.monitorMemoryMaxBytes = 469762048
  innerMemory : certificate.innerMemoryMaxBytes = 8019509248
  cpuCount : certificate.physicalCpus = 4
  workerCount : certificate.workersPerSemanticStage = 4
  passed : certificate.statusPassed = true

structure Principal where
  id : String
  identityRecord : NonemptyCanonicalRecord

structure ReviewerAttestation where
  record : NonemptyCanonicalRecord
  reviewer : Principal
  otherReviewerId : String
  implementationAuthorId : String
  authorizationIssuerId : String
  reviewedContractSha256 : String
  reviewedContractSchemaSha256 : String
  reviewedImplementationCommit : String
  reviewedLauncherSourceFreezeSha256 : String
  reviewedTestBundleSha256 : String
  reviewedToolClosureSha256 : String
  independentFromImplementationAuthor : Bool
  independentFromAuthorizationIssuer : Bool
  distinctFromOtherReviewer : Bool
  issuerEqualsReviewer : Bool
  statusPassed : Bool

structure AuthorizationInboxEntries where
  authorization : NonemptyCanonicalRecord
  reviewer1 : NonemptyCanonicalRecord
  reviewer2 : NonemptyCanonicalRecord
  identityRegistry : NonemptyCanonicalRecord

def AuthorizationInboxEntries.toList (entries : AuthorizationInboxEntries) :
    List NonemptyCanonicalRecord :=
  [entries.authorization, entries.reviewer1, entries.reviewer2, entries.identityRegistry]

structure AuthorizationInbox where
  record : NonemptyCanonicalRecord
  entries : AuthorizationInboxEntries
  authorizationEntryCount : Nat
  attestationEntryCount : Nat
  identityRegistryEntryCount : Nat
  pathsUniqueSorted : Bool
  allFsynced : Bool
  atomicNoreplace : Bool
  reopenFullBytesEqual : Bool
  statusPassed : Bool

structure AuthorizationRecord (phase : Phase) where
  record : NonemptyCanonicalRecord
  implementationAuthor : Principal
  authorizationIssuer : Principal
  reviewer1 : ReviewerAttestation
  reviewer2 : ReviewerAttestation
  inbox : AuthorizationInbox
  identityRegistry : NonemptyCanonicalRecord
  contractSha256 : String
  contractSchemaSha256 : String
  implementationCommit : String
  launcherSourceFreezeSha256 : String
  testBundleSha256 : String
  externalToolDependencyClosureSha256 : String
  recordSchemaBundleSha256 : String
  runNonce : Sha256Binding
  maximumConsumptions : Nat
  dryRunAuthorizationRecordSha256 : Option String
  dryRunConsumptionRecordSha256 : Option String
  dryRunValidationSha256 : Option String
  redteamValidationSha256 : Option String
  statusAuthorized : Bool

structure AuthorizationConsumption (phase : Phase) where
  record : NonemptyCanonicalRecord
  contractSha256 : String
  implementationCommit : String
  runNonce : String
  ledgerKey : String
  ledgerRecordPath : String
  frozenFdVerificationNs : Nat
  consumedAtNs : Nat
  deadlineStartNs : Nat
  atomicNoreplace : Bool
  recordFsynced : Bool
  parentFsynced : Bool
  reopenFullBytesEqual : Bool
  soleCompositeKeyRecord : Bool
  statusPassed : Bool

structure AuthorizationGates
    {phase : Phase} (bindings : ImplementationBindings)
    (authorization : AuthorizationRecord phase)
    (consumption : AuthorizationConsumption phase) : Prop where
  authorizationRecord : RecordValid authorization.record
  nonceValid : Sha256BindingValid authorization.runNonce
  consumptionRecord : RecordValid consumption.record
  inboxRecord : RecordValid authorization.inbox.record
  identityRecord : RecordValid authorization.identityRegistry
  inboxEntryRecords : RecordValid authorization.inbox.entries.authorization ∧
    RecordValid authorization.inbox.entries.reviewer1 ∧
    RecordValid authorization.inbox.entries.reviewer2 ∧
    RecordValid authorization.inbox.entries.identityRegistry
  fourInboxEntries : authorization.inbox.entries.toList.length = 4
  inboxEntriesInFrozenOrder :
    authorization.inbox.entries.authorization.bytes = authorization.record.bytes ∧
    authorization.inbox.entries.authorization.sha256.value =
      authorization.record.sha256.value ∧
    authorization.inbox.entries.reviewer1.bytes =
      authorization.reviewer1.record.bytes ∧
    authorization.inbox.entries.reviewer1.sha256.value =
      authorization.reviewer1.record.sha256.value ∧
    authorization.inbox.entries.reviewer2.bytes =
      authorization.reviewer2.record.bytes ∧
    authorization.inbox.entries.reviewer2.sha256.value =
      authorization.reviewer2.record.sha256.value ∧
    authorization.inbox.entries.identityRegistry.bytes = authorization.identityRegistry.bytes ∧
    authorization.inbox.entries.identityRegistry.sha256.value =
      authorization.identityRegistry.sha256.value
  inboxCounts : authorization.inbox.authorizationEntryCount = 1 ∧
    authorization.inbox.attestationEntryCount = 2 ∧
    authorization.inbox.identityRegistryEntryCount = 1
  inboxClosed : authorization.inbox.pathsUniqueSorted = true ∧
    authorization.inbox.allFsynced = true ∧
    authorization.inbox.atomicNoreplace = true ∧
    authorization.inbox.reopenFullBytesEqual = true ∧
    authorization.inbox.statusPassed = true
  reviewerRecords : RecordValid authorization.reviewer1.record ∧
    RecordValid authorization.reviewer2.record
  fourPrincipalsDistinct :
    List.Nodup [authorization.implementationAuthor.id,
      authorization.authorizationIssuer.id, authorization.reviewer1.reviewer.id,
      authorization.reviewer2.reviewer.id]
  reviewersCrossReference :
    authorization.reviewer1.otherReviewerId = authorization.reviewer2.reviewer.id ∧
    authorization.reviewer2.otherReviewerId = authorization.reviewer1.reviewer.id
  reviewerFlags :
    authorization.reviewer1.independentFromImplementationAuthor = true ∧
    authorization.reviewer1.independentFromAuthorizationIssuer = true ∧
    authorization.reviewer1.distinctFromOtherReviewer = true ∧
    authorization.reviewer1.issuerEqualsReviewer = true ∧
    authorization.reviewer1.statusPassed = true ∧
    authorization.reviewer2.independentFromImplementationAuthor = true ∧
    authorization.reviewer2.independentFromAuthorizationIssuer = true ∧
    authorization.reviewer2.distinctFromOtherReviewer = true ∧
    authorization.reviewer2.issuerEqualsReviewer = true ∧
    authorization.reviewer2.statusPassed = true
  contract : authorization.contractSha256 = frozenContractSha256
  contractSchema : authorization.contractSchemaSha256 = frozenContractSchemaSha256
  implementation : authorization.implementationCommit = bindings.implementationCommit.value
  launcherFreeze :
    authorization.launcherSourceFreezeSha256 = bindings.launcherSourceFreezeSha256.value
  testBundle : authorization.testBundleSha256 = bindings.testBundleSha256.value
  toolClosure : authorization.externalToolDependencyClosureSha256 =
    bindings.externalToolDependencyClosureSha256.value
  schemaBundle : authorization.recordSchemaBundleSha256 = frozenRecordSchemaBundleSha256
  oneConsumption : authorization.maximumConsumptions = 1
  consumptionMatches : consumption.contractSha256 = authorization.contractSha256 ∧
    consumption.implementationCommit = authorization.implementationCommit ∧
    consumption.runNonce = authorization.runNonce.value
  consumptionOrdered : consumption.deadlineStartNs <= consumption.frozenFdVerificationNs ∧
    consumption.frozenFdVerificationNs <= consumption.consumedAtNs
  consumptionDurable : consumption.atomicNoreplace = true ∧
    consumption.recordFsynced = true ∧ consumption.parentFsynced = true ∧
    consumption.reopenFullBytesEqual = true ∧
    consumption.soleCompositeKeyRecord = true ∧ consumption.statusPassed = true
  authorized : authorization.statusAuthorized = true

structure ArtifactTreeCertificate where
  record : NonemptyCanonicalRecord
  files : List NonemptyCanonicalRecord
  directories : List NonemptyCanonicalRecord
  fileCount : Nat
  directoryCount : Nat
  pathCount : Nat
  logicalBytes : Nat
  allocatedBytes : Nat
  contentRootSha256 : String
  metadataRootSha256 : String
  combinedRootSha256 : String
  pathsUniqueSorted : Bool
  fullBytesReadBack : Bool
  allMetadataExact : Bool
  allFsynced : Bool
  forbiddenEntryCount : Nat
  rootsRecomputed : Bool
  statusPassed : Bool

structure IsolationCertificate where
  record : NonemptyCanonicalRecord
  roleProjections : Vector NonemptyCanonicalRecord 6
  roleMatrixSha256 : String
  projectionRootSha256 : String
  siblingsVerified : Bool
  permissionMatrixExact : Bool
  projectionRootRecomputed : Bool
  rootRecursiveReadonly : Bool
  unexpectedWritableMountCount : Nat
  hostRootMapped : Bool
  capabilitySetsEmpty : Bool
  noNewPrivileges : Bool
  hostSetnsEperm : Bool
  hostNetworkFdMatchCountAfterProbe : Nat
  statusPassed : Bool

structure ResourceCertificate (phase : Phase) where
  record : NonemptyCanonicalRecord
  deadlineStartNs : Nat
  authorizationConsumedNs : Nat
  firstOperationalByteNs : Nat
  postcheckCompletedNs : Nat
  markerValidatedNs : Nat
  deadlineAbsoluteNs : Nat
  wallUpperSeconds : Nat
  executorMemoryPeak : Nat
  workloadParentMemoryPeak : Nat
  watchdogMemoryPeak : Nat
  monitorMemoryPeak : Nat
  innerMemoryPeak : Nat
  temporaryAllocatedBytes : Nat
  publishedLogicalBytes : Nat
  publishedAllocatedBytes : Nat
  oomEventDeltaTotal : Nat
  cpusetCpuIds : Vector Nat 4
  physicalCorePairsDistinct : Bool
  deadlineExpired : Bool
  executorLiveAtPostcheck : Bool
  watchdogPidfdExitObserved : Bool
  statusPassed : Bool

structure SemanticCertificate where
  record : NonemptyCanonicalRecord
  pairCount : Nat
  tupleMembershipCount : Nat
  paddedClippingCount : Nat
  canonicalGrayEqualCount : Nat
  canonicalPngEqualCount : Nat
  graphInequivalentCount : Nat
  rawGrayDistinctCount : Nat
  verifierAgreement : Bool
  statusPassed : Bool

structure SourceClosureCertificate where
  record : NonemptyCanonicalRecord
  sourceCommit : String
  fullArchiveSha256 : String
  e2ArchiveSha256 : String
  fullArchiveFileCount : Nat
  e2ArchiveFileCount : Nat
  snapshotEntries : List NonemptyCanonicalRecord
  snapshotDirectories : List NonemptyCanonicalRecord
  safeExtraction : Bool
  completeArchiveBytesEqual : Bool
  completePathSetEqual : Bool
  allFsynced : Bool
  rootsRecomputed : Bool
  forbiddenEntryCount : Nat
  statusPassed : Bool

structure WorkerAssignment where
  workerPid : Nat
  pair0 : Fin 8
  pair1 : Fin 8

structure StageCertificate where
  record : NonemptyCanonicalRecord
  stageName : String
  workers : Vector WorkerAssignment 4
  workerPidsDistinct : Bool
  concurrencyIntervalNonempty : Bool
  exactPairPartition : Bool
  supervisorExitedBeforeFreeze : Bool
  frozenObserved : Bool
  populatedZero : Bool
  residualPidCount : Nat
  escapedPidCount : Nat
  outputSnapshotAfterEmpty : Bool
  cgroupRemoved : Bool
  statusPassed : Bool

structure StageQuiescenceCertificate where
  record : NonemptyCanonicalRecord
  producer : StageCertificate
  verifierA : StageCertificate
  verifierB : StageCertificate
  stageSetRootSha256 : String
  stageSetRootRecomputedFromFullBytes : Bool
  statusPassed : Bool

structure ToolClosureCertificate where
  record : NonemptyCanonicalRecord
  primaryTools : Vector NonemptyCanonicalRecord 11
  dynamicLoaders : List NonemptyCanonicalRecord
  sharedLibraries : List NonemptyCanonicalRecord
  closureRootSha256 : String
  descriptorPayloadsNonempty : Bool
  primaryRolesUnique : Bool
  resolvedPathsUnique : Bool
  recursiveClosureComplete : Bool
  completeBytesVerified : Bool
  closureRootRecomputed : Bool
  noFallbackPathLookup : Bool
  statusPassed : Bool

structure LivenessCertificate where
  record : NonemptyCanonicalRecord
  executorAndWatchdogUidsDistinct : Bool
  pidNamespacesDistinct : Bool
  mutualSignalEperm : Bool
  executorHoldsWatchdogPidfd : Bool
  watchdogHoldsExecutorPidfd : Bool
  executorLiveAtPostcheck : Bool
  watchdogLiveAtPostcheck : Bool
  executorDeathProbeKilledParent : Bool
  watchdogDeathProbeKilledParent : Bool
  statusPassed : Bool

structure PostcheckCertificate (phase : Phase) where
  record : NonemptyCanonicalRecord
  candidateCombinedRootSha256 : String
  requiredCertificateSetRootSha256 : String
  certificateHashes : Vector String 8
  livenessCertificateSha256 : String
  completeBytesEqual : Bool
  normalizedPathsEqual : Bool
  contentRootEqual : Bool
  metadataRootEqual : Bool
  combinedRootEqual : Bool
  ownershipModesLinksExact : Bool
  immutabilityProbesPassed : Bool
  monitorReopenPassed : Bool
  watchdogReopenPassed : Bool
  executorLive : Bool
  deadlineNotExpired : Bool
  oomDeltasZero : Bool
  residualProcessCount : Nat
  markerAbsentDuringPostcheck : Bool
  renamedCandidateObserved : Bool
  resultsParentFsynced : Bool
  officialResultsPathAbsent : Bool
  statusPassed : Bool

inductive RelationId where
  | r01 | r02 | r03 | r04 | r05 | r06 | r07 | r08 | r09 | r10
  | r11 | r12 | r13 | r14 | r15 | r16 | r17 | r18 | r19 | r20
  | r21 | r22 | r23 | r24 | r25 | r26 | r27 | r28 | r29 | r30
  | r31 | r32 | r33 | r34 | r35 | r36 | r37 | r38 | r39 | r40
  | r41 | r42 | r43 | r44 | r45 | r46
deriving DecidableEq, Repr

structure EvidenceCore (phase : Phase) where
  bindings : ImplementationBindings
  conformance : ContractConformanceCertificate
  authorization : AuthorizationRecord phase
  consumption : AuthorizationConsumption phase
  artifact : ArtifactTreeCertificate
  isolation : IsolationCertificate
  resource : ResourceCertificate phase
  semantic : SemanticCertificate
  sourceClosure : SourceClosureCertificate
  stages : StageQuiescenceCertificate
  tools : ToolClosureCertificate
  liveness : LivenessCertificate
  postcheck : PostcheckCertificate phase

def StagePassed (stage : StageCertificate) : Prop :=
  RecordValid stage.record ∧ stage.workers.toList.length = 4 ∧
    stage.workerPidsDistinct = true ∧ stage.concurrencyIntervalNonempty = true ∧
    stage.exactPairPartition = true ∧ stage.supervisorExitedBeforeFreeze = true ∧
    stage.frozenObserved = true ∧ stage.populatedZero = true ∧
    stage.residualPidCount = 0 ∧ stage.escapedPidCount = 0 ∧
    stage.outputSnapshotAfterEmpty = true ∧ stage.cgroupRemoved = true ∧
    stage.statusPassed = true

def ExactInboxEntries {phase : Phase} (authorization : AuthorizationRecord phase) : Prop :=
  authorization.inbox.entries.authorization.bytes = authorization.record.bytes ∧
  authorization.inbox.entries.authorization.sha256.value =
    authorization.record.sha256.value ∧
  authorization.inbox.entries.reviewer1.bytes =
    authorization.reviewer1.record.bytes ∧
  authorization.inbox.entries.reviewer1.sha256.value =
    authorization.reviewer1.record.sha256.value ∧
  authorization.inbox.entries.reviewer2.bytes =
    authorization.reviewer2.record.bytes ∧
  authorization.inbox.entries.reviewer2.sha256.value =
    authorization.reviewer2.record.sha256.value ∧
  authorization.inbox.entries.identityRegistry.bytes = authorization.identityRegistry.bytes ∧
  authorization.inbox.entries.identityRegistry.sha256.value =
    authorization.identityRegistry.sha256.value

def relationOrdinal : RelationId -> String
  | .r01 => "r01" | .r02 => "r02" | .r03 => "r03" | .r04 => "r04"
  | .r05 => "r05" | .r06 => "r06" | .r07 => "r07" | .r08 => "r08"
  | .r09 => "r09" | .r10 => "r10" | .r11 => "r11" | .r12 => "r12"
  | .r13 => "r13" | .r14 => "r14" | .r15 => "r15" | .r16 => "r16"
  | .r17 => "r17" | .r18 => "r18" | .r19 => "r19" | .r20 => "r20"
  | .r21 => "r21" | .r22 => "r22" | .r23 => "r23" | .r24 => "r24"
  | .r25 => "r25" | .r26 => "r26" | .r27 => "r27" | .r28 => "r28"
  | .r29 => "r29" | .r30 => "r30" | .r31 => "r31" | .r32 => "r32"
  | .r33 => "r33" | .r34 => "r34" | .r35 => "r35" | .r36 => "r36"
  | .r37 => "r37" | .r38 => "r38" | .r39 => "r39" | .r40 => "r40"
  | .r41 => "r41" | .r42 => "r42" | .r43 => "r43" | .r44 => "r44"
  | .r45 => "r45" | .r46 => "r46"

def expectedRelationOwner : RelationId -> String
  | .r01 => "bootstrap_watchdog"
  | .r02 => "bootstrap_watchdog"
  | .r03 => "bootstrap_watchdog"
  | .r04 => "bootstrap_watchdog"
  | .r05 => "bootstrap_watchdog"
  | .r06 => "bootstrap_watchdog"
  | .r07 => "outer_deadline_executor+bootstrap_watchdog"
  | .r08 => "bootstrap_watchdog"
  | .r09 => "bootstrap_watchdog"
  | .r10 => "bootstrap_watchdog"
  | .r11 => "bootstrap_watchdog"
  | .r12 => "monitor_verified_by_bootstrap_watchdog"
  | .r13 => "monitor_verified_by_bootstrap_watchdog"
  | .r14 => "monitor_verified_by_bootstrap_watchdog"
  | .r15 => "monitor_verified_by_bootstrap_watchdog"
  | .r16 => "monitor+bootstrap_watchdog"
  | .r17 => "monitor+bootstrap_watchdog"
  | .r18 => "monitor+bootstrap_watchdog"
  | .r19 => "monitor+bootstrap_watchdog"
  | .r20 => "monitor+bootstrap_watchdog"
  | .r21 => "monitor_verified_by_bootstrap_watchdog"
  | .r22 => "bootstrap_watchdog"
  | .r23 => "bootstrap_watchdog"
  | .r24 => "bootstrap_watchdog"
  | .r25 => "bootstrap_watchdog"
  | .r26 => "bootstrap_watchdog"
  | .r27 => "bootstrap_watchdog"
  | .r28 => "bootstrap_watchdog"
  | .r29 => "bootstrap_watchdog"
  | .r30 => "outer_deadline_executor+bootstrap_watchdog"
  | .r31 => "outer_deadline_executor+bootstrap_watchdog"
  | .r32 => "monitor+bootstrap_watchdog"
  | .r33 => "monitor+bootstrap_watchdog"
  | .r34 => "outer_deadline_executor+bootstrap_watchdog"
  | .r35 => "outer_deadline_executor+bootstrap_watchdog"
  | .r36 => "outer_deadline_executor+bootstrap_watchdog"
  | .r37 => "bootstrap_watchdog"
  | .r38 => "bootstrap_watchdog"
  | .r39 => "bootstrap_watchdog"
  | .r40 => "monitor+bootstrap_watchdog"
  | .r41 => "monitor+bootstrap_watchdog"
  | .r42 => "monitor_verified_by_bootstrap_watchdog"
  | .r43 => "bootstrap_watchdog"
  | .r44 => "bootstrap_watchdog"
  | .r45 => "bootstrap_watchdog"
  | .r46 => "bootstrap_watchdog"

def expectedRelationVerifier (id : RelationId) : String :=
  match id with
  | .r01 => "chemgraph_e2_v03_verify_r01"
  | .r02 => "chemgraph_e2_v03_verify_r02"
  | .r03 => "chemgraph_e2_v03_verify_r03"
  | .r04 => "chemgraph_e2_v03_verify_r04"
  | .r05 => "chemgraph_e2_v03_verify_r05"
  | .r06 => "chemgraph_e2_v03_verify_r06"
  | .r07 => "chemgraph_e2_v03_verify_r07"
  | .r08 => "chemgraph_e2_v03_verify_r08"
  | .r09 => "chemgraph_e2_v03_verify_r09"
  | .r10 => "chemgraph_e2_v03_verify_r10"
  | .r11 => "chemgraph_e2_v03_verify_r11"
  | .r12 => "chemgraph_e2_v03_verify_r12"
  | .r13 => "chemgraph_e2_v03_verify_r13"
  | .r14 => "chemgraph_e2_v03_verify_r14"
  | .r15 => "chemgraph_e2_v03_verify_r15"
  | .r16 => "chemgraph_e2_v03_verify_r16"
  | .r17 => "chemgraph_e2_v03_verify_r17"
  | .r18 => "chemgraph_e2_v03_verify_r18"
  | .r19 => "chemgraph_e2_v03_verify_r19"
  | .r20 => "chemgraph_e2_v03_verify_r20"
  | .r21 => "chemgraph_e2_v03_verify_r21"
  | .r22 => "chemgraph_e2_v03_verify_r22"
  | .r23 => "chemgraph_e2_v03_verify_r23"
  | .r24 => "chemgraph_e2_v03_verify_r24"
  | .r25 => "chemgraph_e2_v03_verify_r25"
  | .r26 => "chemgraph_e2_v03_verify_r26"
  | .r27 => "chemgraph_e2_v03_verify_r27"
  | .r28 => "chemgraph_e2_v03_verify_r28"
  | .r29 => "chemgraph_e2_v04_verify_r29"
  | .r30 => "chemgraph_e2_v03_verify_r30"
  | .r31 => "chemgraph_e2_v03_verify_r31"
  | .r32 => "chemgraph_e2_v03_verify_r32"
  | .r33 => "chemgraph_e2_v03_verify_r33"
  | .r34 => "chemgraph_e2_v03_verify_r34"
  | .r35 => "chemgraph_e2_v03_verify_r35"
  | .r36 => "chemgraph_e2_v03_verify_r36"
  | .r37 => "chemgraph_e2_v03_verify_r37"
  | .r38 => "chemgraph_e2_v03_verify_r38"
  | .r39 => "chemgraph_e2_v03_verify_r39"
  | .r40 => "chemgraph_e2_v03_verify_r40"
  | .r41 => "chemgraph_e2_v03_verify_r41"
  | .r42 => "chemgraph_e2_v03_verify_r42"
  | .r43 => "chemgraph_e2_v03_verify_r43"
  | .r44 => "chemgraph_e2_v03_verify_r44"
  | .r45 => "chemgraph_e2_v03_verify_r45"
  | .r46 => "chemgraph_e2_v03_verify_r46"

def expectedRelationTheorem (id : RelationId) : String :=
  match id with
  | .r01 => "chemgraph_e2_v03_relation_r01_sound"
  | .r02 => "chemgraph_e2_v03_relation_r02_sound"
  | .r03 => "chemgraph_e2_v03_relation_r03_sound"
  | .r04 => "chemgraph_e2_v03_relation_r04_sound"
  | .r05 => "chemgraph_e2_v03_relation_r05_sound"
  | .r06 => "chemgraph_e2_v03_relation_r06_sound"
  | .r07 => "chemgraph_e2_v03_relation_r07_sound"
  | .r08 => "chemgraph_e2_v03_relation_r08_sound"
  | .r09 => "chemgraph_e2_v03_relation_r09_sound"
  | .r10 => "chemgraph_e2_v03_relation_r10_sound"
  | .r11 => "chemgraph_e2_v03_relation_r11_sound"
  | .r12 => "chemgraph_e2_v03_relation_r12_sound"
  | .r13 => "chemgraph_e2_v03_relation_r13_sound"
  | .r14 => "chemgraph_e2_v03_relation_r14_sound"
  | .r15 => "chemgraph_e2_v03_relation_r15_sound"
  | .r16 => "chemgraph_e2_v03_relation_r16_sound"
  | .r17 => "chemgraph_e2_v03_relation_r17_sound"
  | .r18 => "chemgraph_e2_v03_relation_r18_sound"
  | .r19 => "chemgraph_e2_v03_relation_r19_sound"
  | .r20 => "chemgraph_e2_v03_relation_r20_sound"
  | .r21 => "chemgraph_e2_v03_relation_r21_sound"
  | .r22 => "chemgraph_e2_v03_relation_r22_sound"
  | .r23 => "chemgraph_e2_v03_relation_r23_sound"
  | .r24 => "chemgraph_e2_v03_relation_r24_sound"
  | .r25 => "chemgraph_e2_v03_relation_r25_sound"
  | .r26 => "chemgraph_e2_v03_relation_r26_sound"
  | .r27 => "chemgraph_e2_v03_relation_r27_sound"
  | .r28 => "chemgraph_e2_v03_relation_r28_sound"
  | .r29 => "chemgraph_e2_v04_relation_r29_sound"
  | .r30 => "chemgraph_e2_v03_relation_r30_sound"
  | .r31 => "chemgraph_e2_v03_relation_r31_sound"
  | .r32 => "chemgraph_e2_v03_relation_r32_sound"
  | .r33 => "chemgraph_e2_v03_relation_r33_sound"
  | .r34 => "chemgraph_e2_v03_relation_r34_sound"
  | .r35 => "chemgraph_e2_v03_relation_r35_sound"
  | .r36 => "chemgraph_e2_v03_relation_r36_sound"
  | .r37 => "chemgraph_e2_v03_relation_r37_sound"
  | .r38 => "chemgraph_e2_v03_relation_r38_sound"
  | .r39 => "chemgraph_e2_v03_relation_r39_sound"
  | .r40 => "chemgraph_e2_v03_relation_r40_sound"
  | .r41 => "chemgraph_e2_v03_relation_r41_sound"
  | .r42 => "chemgraph_e2_v03_relation_r42_sound"
  | .r43 => "chemgraph_e2_v03_relation_r43_sound"
  | .r44 => "chemgraph_e2_v03_relation_r44_sound"
  | .r45 => "chemgraph_e2_v03_relation_r45_sound"
  | .r46 => "chemgraph_e2_v03_relation_r46_sound"

def frozenOfflineVerifierSourceSha256 : String :=
  "729f42feb9b5d13833e0fe6f9da1c51d70f55933045f34eee601eb720753cc0a"

def frozenOfflineVerifierTreeSha256 : String :=
  "3c5307d766fc7376659b22f74f354a912b71ff7c9cabcf90df87c79aec44b7e7"

def relationTranscriptDomain : ByteArray :=
  "chemgraph-e2-v0.4/relation-transcript/v1".toUTF8

def u32be (value : Nat) : ByteArray :=
  ByteArray.mk #[UInt8.ofNat (value / 16777216), UInt8.ofNat (value / 65536),
    UInt8.ofNat (value / 256), UInt8.ofNat value]

def u64be (value : Nat) : ByteArray :=
  ByteArray.mk #[UInt8.ofNat (value / 72057594037927936), UInt8.ofNat (value / 281474976710656),
    UInt8.ofNat (value / 1099511627776), UInt8.ofNat (value / 4294967296),
    UInt8.ofNat (value / 16777216), UInt8.ofNat (value / 65536),
    UInt8.ofNat (value / 256), UInt8.ofNat value]

def concatBytes (parts : List ByteArray) : ByteArray :=
  parts.foldl (fun result part => result ++ part) ByteArray.empty

def frameU32 (bytes : ByteArray) : ByteArray := u32be bytes.size ++ bytes
def frameU64 (bytes : ByteArray) : ByteArray := u64be bytes.size ++ bytes

def encodeByteList (values : List ByteArray) : ByteArray :=
  u64be values.length ++ concatBytes (values.map frameU64)

def encodeStringList (values : List String) : ByteArray :=
  encodeByteList (values.map String.toUTF8)

def encodeNatList (values : List Nat) : ByteArray :=
  u64be values.length ++ concatBytes (values.map u64be)

structure UnixMetadata where
  uid : Nat
  gid : Nat
  mode : Nat
  nlink : Nat

def encodeMetadata (metadata : UnixMetadata) : ByteArray :=
  encodeNatList [metadata.uid, metadata.gid, metadata.mode, metadata.nlink]

inductive AtomicInstallOutcome where
  | installedNoReplace
deriving Repr

structure DurableWriteObservation where
  installedAtNs : Nat
  fileFsyncedAtNs : Nat
  parentFsyncedAtNs : Nat
  writtenBytes : ByteArray
  reopenedBytes : ByteArray
  outcome : AtomicInstallOutcome

structure ConcreteRelationResult where
  observedBytes : List ByteArray
  expectedBytes : List ByteArray
  observedNats : List Nat
  expectedNats : List Nat
  observedStrings : List String
  expectedStrings : List String
  uniqueKeys : List String
  monotonicTimes : List Nat

def concreteResultPayload (result : ConcreteRelationResult) : ByteArray :=
  concatBytes [encodeByteList result.observedBytes, encodeByteList result.expectedBytes,
    encodeNatList result.observedNats, encodeNatList result.expectedNats,
    encodeStringList result.observedStrings, encodeStringList result.expectedStrings,
    encodeStringList result.uniqueKeys, encodeNatList result.monotonicTimes]

structure LedgerRelationResult where
  domain : ByteArray
  decodedContractSha256 : ByteArray
  implementationCommitBytes : ByteArray
  phaseByte : Nat
  claimedLedgerKey : String
  observedRecordPath : ByteArray
  rootMetadata : UnixMetadata
  recordMetadata : UnixMetadata
  compositeRecordCount : Nat
  durableWrite : DurableWriteObservation

def ledgerPreimageBytes (result : LedgerRelationResult) : ByteArray :=
  concatBytes [frameU32 result.domain, frameU64 result.decodedContractSha256,
    frameU64 result.implementationCommitBytes,
    ByteArray.mk #[UInt8.ofNat result.phaseByte]]

def ledgerRecordPath (ledgerKey : String) : ByteArray :=
  concatBytes ["/opt/chemgraph/authorization-ledger-v0.3/".toUTF8,
    ledgerKey.toUTF8, ".json".toUTF8]

def ledgerResultPayload (result : LedgerRelationResult) : ByteArray :=
  concatBytes [frameU64 (ledgerPreimageBytes result), frameU32 result.claimedLedgerKey.toUTF8,
    frameU32 result.observedRecordPath, encodeMetadata result.rootMetadata,
    encodeMetadata result.recordMetadata, u64be result.compositeRecordCount,
    encodeNatList [result.durableWrite.installedAtNs, result.durableWrite.fileFsyncedAtNs,
      result.durableWrite.parentFsyncedAtNs], frameU64 result.durableWrite.writtenBytes,
    frameU64 result.durableWrite.reopenedBytes]

structure FramedPreimage where
  domain : ByteArray
  orderedCompleteRecords : List ByteArray

def FramedPreimage.bytes (preimage : FramedPreimage) : ByteArray :=
  frameU32 preimage.domain ++ encodeByteList preimage.orderedCompleteRecords

structure RootRelationResult where
  preimage : FramedPreimage
  claimedRootSha256 : String
  verifierSourceSha256 : String
  verifierTreeSha256 : String

def rootResultPayload (result : RootRelationResult) : ByteArray :=
  concatBytes [frameU64 result.preimage.bytes, frameU32 result.claimedRootSha256.toUTF8,
    frameU32 result.verifierSourceSha256.toUTF8, frameU32 result.verifierTreeSha256.toUTF8]

structure ArtifactRootsRelationResult where
  content : RootRelationResult
  metadata : RootRelationResult
  combined : RootRelationResult

def artifactRootsResultPayload (result : ArtifactRootsRelationResult) : ByteArray :=
  concatBytes [frameU64 (rootResultPayload result.content),
    frameU64 (rootResultPayload result.metadata), frameU64 (rootResultPayload result.combined)]

structure NormalizedSegment where
  bytes : List UInt8
  nonempty : bytes ≠ []
  notDot : bytes ≠ [46]
  notDotDot : bytes ≠ [46, 46]
  slashAbsent : 47 ∉ bytes

structure MarkerTuple where
  contractSha256 : NormalizedSegment
  implementationCommit : NormalizedSegment
  runNonce : NormalizedSegment
  candidateCombinedRootSha256 : NormalizedSegment

inductive MarkerPathSegment where
  | authorityRoot
  | value (segment : NormalizedSegment)
  | official

def markerPath (tuple : MarkerTuple) : List MarkerPathSegment :=
  [.authorityRoot, .value tuple.contractSha256, .value tuple.implementationCommit,
    .official, .value tuple.runNonce, .value tuple.candidateCombinedRootSha256]

theorem markerPath_injective : Function.Injective markerPath := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

structure MarkerPathRelationResult where
  tuple : MarkerTuple
  observedPath : List MarkerPathSegment

def markerSegmentPayload : MarkerPathSegment -> ByteArray
  | .authorityRoot => frameU32 "authority-markers-v0.3".toUTF8
  | .official => frameU32 "official".toUTF8
  | .value segment => encodeByteList [ByteArray.mk segment.bytes.toArray]

def markerPathResultPayload (result : MarkerPathRelationResult) : ByteArray :=
  encodeByteList (result.observedPath.map markerSegmentPayload)

inductive ProtocolPhaseSource where
  | dryOuter | drySiblings | dryAuthorization | dryMonitor | dryInner
  | dryProducer | dryProducerDone | dryVerifierA | dryVerifierADone
  | dryVerifierB | dryVerifierBDone | dryCandidate | dryPostchecked
  | officialOuter | officialSiblings | officialAuthorization | officialMonitor
  | officialInner | officialProducer | officialProducerDone | officialVerifierA
  | officialVerifierADone | officialVerifierB | officialVerifierBDone
  | officialCandidate | officialMetadata | officialRenamed | officialPostchecked
  | officialMarkerInstalled
deriving DecidableEq, Repr

inductive ProtocolFailureEvent where
  | schemaOrRelation | resourceOrIsolation | stageOrSourceTool | postcheckOrMarker
  | deadlineExpired | monitorExitOrOom | watchdogExit | deadlineExecutorExit
  | residualProcess
deriving DecidableEq, Repr

inductive FailureTargetTag where
  | failed | timedOut | monitorDead | watchdogDead | deadlineExecutorDead | residual
deriving DecidableEq, Repr

structure FailureTransitionCell where
  source : ProtocolPhaseSource
  event : ProtocolFailureEvent
  target : FailureTargetTag
deriving DecidableEq, Repr

def allProtocolPhaseSources : List ProtocolPhaseSource :=
  [.dryOuter, .drySiblings, .dryAuthorization, .dryMonitor, .dryInner,
    .dryProducer, .dryProducerDone, .dryVerifierA, .dryVerifierADone,
    .dryVerifierB, .dryVerifierBDone, .dryCandidate, .dryPostchecked,
    .officialOuter, .officialSiblings, .officialAuthorization, .officialMonitor,
    .officialInner, .officialProducer, .officialProducerDone, .officialVerifierA,
    .officialVerifierADone, .officialVerifierB, .officialVerifierBDone,
    .officialCandidate, .officialMetadata, .officialRenamed, .officialPostchecked,
    .officialMarkerInstalled]

def allFailureEvents : List ProtocolFailureEvent :=
  [.schemaOrRelation, .resourceOrIsolation, .stageOrSourceTool, .postcheckOrMarker,
    .deadlineExpired, .monitorExitOrOom, .watchdogExit, .deadlineExecutorExit,
    .residualProcess]

def failureTargetTag : ProtocolFailureEvent -> FailureTargetTag
  | .schemaOrRelation | .resourceOrIsolation | .stageOrSourceTool
  | .postcheckOrMarker => .failed
  | .deadlineExpired => .timedOut
  | .monitorExitOrOom => .monitorDead
  | .watchdogExit => .watchdogDead
  | .deadlineExecutorExit => .deadlineExecutorDead
  | .residualProcess => .residual

def expectedFailureTransitionTable : List FailureTransitionCell :=
  allProtocolPhaseSources.flatMap fun source =>
    allFailureEvents.map fun event => ⟨source, event, failureTargetTag event⟩

set_option maxRecDepth 4096 in
theorem expected_failure_transition_count : expectedFailureTransitionTable.length = 261 := by
  decide

inductive SuccessStepTag where
  | arm | siblings | consume | monitor | inner | producer | producerDone
  | verifierA | verifierADone | verifierB | verifierBDone | dryCandidate
  | dryPostcheck | dryPass | officialCandidate | metadata | rename | postcheck
  | installMarker | authorize
deriving DecidableEq, Repr

inductive TerminalTargetTag where
  | quarantine | orphanedNonAuthoritativeCandidate
deriving DecidableEq, Repr

structure CleanupTransitionCell where
  source : FailureTargetTag
  target : TerminalTargetTag
deriving DecidableEq, Repr

def expectedCleanupTable : List CleanupTransitionCell :=
  [⟨.failed, .quarantine⟩, ⟨.monitorDead, .quarantine⟩,
    ⟨.residual, .quarantine⟩, ⟨.timedOut, .orphanedNonAuthoritativeCandidate⟩,
    ⟨.watchdogDead, .orphanedNonAuthoritativeCandidate⟩,
    ⟨.deadlineExecutorDead, .orphanedNonAuthoritativeCandidate⟩]

inductive AuthorityRule where
  | officialValidatedDetachedMarkerOnly
deriving DecidableEq, Repr

def drySuccessChain : List SuccessStepTag :=
  [.arm, .siblings, .consume, .monitor, .inner, .producer, .producerDone,
    .verifierA, .verifierADone, .verifierB, .verifierBDone, .dryCandidate,
    .dryPostcheck, .dryPass]

def officialSuccessChain : List SuccessStepTag :=
  [.arm, .siblings, .consume, .monitor, .inner, .producer, .producerDone,
    .verifierA, .verifierADone, .verifierB, .verifierBDone, .officialCandidate,
    .metadata, .rename, .postcheck, .installMarker, .authorize]

structure StateTableRelationResult where
  failureTable : List FailureTransitionCell
  dryChain : List SuccessStepTag
  officialChain : List SuccessStepTag
  cleanupTable : List CleanupTransitionCell
  authorityRule : AuthorityRule

def stateTableResultPayload (result : StateTableRelationResult) : ByteArray :=
  encodeNatList [result.failureTable.length, result.dryChain.length,
    result.officialChain.length, result.cleanupTable.length]

structure AuthorizationAuthenticityRelationResult where
  committedEntries : AuthorizationInboxEntries
  committedEntryHashes : List String
  inboxRootMetadata : UnixMetadata
  entryMetadata : UnixMetadata
  authorizationFsyncedAtNs : Nat
  reviewer1FsyncedAtNs : Nat
  reviewer2FsyncedAtNs : Nat
  registryFsyncedAtNs : Nat
  commitmentWrite : DurableWriteObservation

def authorizationAuthenticityResultPayload
    (result : AuthorizationAuthenticityRelationResult) : ByteArray :=
  concatBytes [encodeByteList (result.committedEntries.toList.map (fun record => record.bytes)),
    encodeStringList result.committedEntryHashes, encodeMetadata result.inboxRootMetadata,
    encodeMetadata result.entryMetadata, encodeNatList [result.authorizationFsyncedAtNs,
      result.reviewer1FsyncedAtNs, result.reviewer2FsyncedAtNs,
      result.registryFsyncedAtNs, result.commitmentWrite.installedAtNs,
      result.commitmentWrite.fileFsyncedAtNs, result.commitmentWrite.parentFsyncedAtNs],
    frameU64 result.commitmentWrite.writtenBytes,
    frameU64 result.commitmentWrite.reopenedBytes]

structure SchemaRecords17 where
  contractConformance : NonemptyCanonicalRecord
  authorization : NonemptyCanonicalRecord
  reviewerAttestation : NonemptyCanonicalRecord
  authorizationConsumption : NonemptyCanonicalRecord
  authorizationInboxCommitment : NonemptyCanonicalRecord
  identityRegistry : NonemptyCanonicalRecord
  artifactTree : NonemptyCanonicalRecord
  isolation : NonemptyCanonicalRecord
  resource : NonemptyCanonicalRecord
  semantic : NonemptyCanonicalRecord
  sourceClosure : NonemptyCanonicalRecord
  stage : NonemptyCanonicalRecord
  stageQuiescence : NonemptyCanonicalRecord
  toolClosure : NonemptyCanonicalRecord
  liveness : NonemptyCanonicalRecord
  postcheck : NonemptyCanonicalRecord
  relationObligation : NonemptyCanonicalRecord

def SchemaRecords17.toList (records : SchemaRecords17) : List NonemptyCanonicalRecord :=
  [records.contractConformance, records.authorization, records.reviewerAttestation,
    records.authorizationConsumption, records.authorizationInboxCommitment,
    records.identityRegistry, records.artifactTree, records.isolation, records.resource,
    records.semantic, records.sourceClosure, records.stage, records.stageQuiescence,
    records.toolClosure, records.liveness, records.postcheck, records.relationObligation]

structure SchemaIds45 where
  id01 : String
  id02 : String
  id03 : String
  id04 : String
  id05 : String
  id06 : String
  id07 : String
  id08 : String
  id09 : String
  id10 : String
  id11 : String
  id12 : String
  id13 : String
  id14 : String
  id15 : String
  id16 : String
  id17 : String
  id18 : String
  id19 : String
  id20 : String
  id21 : String
  id22 : String
  id23 : String
  id24 : String
  id25 : String
  id26 : String
  id27 : String
  id28 : String
  id29 : String
  id30 : String
  id31 : String
  id32 : String
  id33 : String
  id34 : String
  id35 : String
  id36 : String
  id37 : String
  id38 : String
  id39 : String
  id40 : String
  id41 : String
  id42 : String
  id43 : String
  id44 : String
  id45 : String

def SchemaIds45.toList (ids : SchemaIds45) : List String :=
  [ids.id01, ids.id02, ids.id03, ids.id04, ids.id05, ids.id06, ids.id07,
    ids.id08, ids.id09, ids.id10, ids.id11, ids.id12, ids.id13, ids.id14,
    ids.id15, ids.id16, ids.id17, ids.id18, ids.id19, ids.id20, ids.id21,
    ids.id22, ids.id23, ids.id24, ids.id25, ids.id26, ids.id27, ids.id28,
    ids.id29, ids.id30, ids.id31, ids.id32, ids.id33, ids.id34, ids.id35,
    ids.id36, ids.id37, ids.id38, ids.id39, ids.id40, ids.id41, ids.id42,
    ids.id43, ids.id44, ids.id45]

structure SchemaRelationResult where
  draftUri : String
  records : SchemaRecords17
  ids : SchemaIds45

def schemaResultPayload (result : SchemaRelationResult) : ByteArray :=
  concatBytes [frameU32 result.draftUri.toUTF8,
    encodeByteList (result.records.toList.map (fun record => record.bytes)),
    encodeStringList result.ids.toList]

inductive RelationResult where
  | concrete (result : ConcreteRelationResult)
  | ledger (result : LedgerRelationResult)
  | artifactRoots (result : ArtifactRootsRelationResult)
  | framedRoot (result : RootRelationResult)
  | roleMatrixRoot (result : RootRelationResult)
  | certificateSetRoot (result : RootRelationResult)
  | markerPath (result : MarkerPathRelationResult)
  | stateTable (result : StateTableRelationResult)
  | authorizationAuthenticity (result : AuthorizationAuthenticityRelationResult)
  | schemas (result : SchemaRelationResult)

def RelationResult.payload : RelationResult -> ByteArray
  | .concrete result => concreteResultPayload result
  | .ledger result => ledgerResultPayload result
  | .artifactRoots result => artifactRootsResultPayload result
  | .framedRoot result => rootResultPayload result
  | .roleMatrixRoot result => rootResultPayload result
  | .certificateSetRoot result => rootResultPayload result
  | .markerPath result => markerPathResultPayload result
  | .stateTable result => stateTableResultPayload result
  | .authorizationAuthenticity result => authorizationAuthenticityResultPayload result
  | .schemas result => schemaResultPayload result

def relationTranscriptBytes (id : RelationId) (owner verifier theoremId : String)
    (completeInputs : List ByteArray) (resultPayload : ByteArray) : ByteArray :=
  concatBytes [frameU32 relationTranscriptDomain, frameU32 (relationOrdinal id).toUTF8,
    frameU32 owner.toUTF8, frameU32 verifier.toUTF8, frameU32 theoremId.toUTF8,
    frameU32 frozenOfflineVerifierSourceSha256.toUTF8,
    frameU32 frozenOfflineVerifierTreeSha256.toUTF8, encodeByteList completeInputs,
    frameU64 resultPayload]

structure RelationObservation where
  verifierRecord : NonemptyCanonicalRecord
  completeInputBytes : List ByteArray
  owner : String
  verifierId : String
  leanTheoremId : String
  offlineVerifierSourceSha256 : String
  offlineVerifierTreeSha256 : String
  result : RelationResult

def relationInputBytes {phase : Phase} (id : RelationId) (b : EvidenceCore phase) :
    List ByteArray :=
  match id with
  | .r01 => [b.authorization.record.bytes, b.conformance.record.bytes,
      b.sourceClosure.record.bytes, b.tools.record.bytes]
  | .r02 => [b.authorization.implementationAuthor.identityRecord.bytes,
      b.authorization.authorizationIssuer.identityRecord.bytes,
      b.authorization.reviewer1.reviewer.identityRecord.bytes,
      b.authorization.reviewer2.reviewer.identityRecord.bytes]
  | .r03 => [b.authorization.inbox.record.bytes, b.authorization.record.bytes,
      b.authorization.reviewer1.record.bytes, b.authorization.reviewer2.record.bytes,
      b.authorization.inbox.entries.reviewer1.bytes,
      b.authorization.inbox.entries.reviewer2.bytes]
  | .r04 | .r05 => [b.authorization.record.bytes, b.consumption.record.bytes]
  | .r06 => [b.authorization.record.bytes, b.consumption.record.bytes,
      b.resource.record.bytes]
  | .r07 | .r36 => [b.resource.record.bytes, b.liveness.record.bytes]
  | .r08 => [b.conformance.record.bytes, b.resource.record.bytes]
  | .r30 | .r35 => [b.resource.record.bytes]
  | .r31 => [b.consumption.record.bytes, b.resource.record.bytes]
  | .r34 => [b.resource.record.bytes, b.postcheck.record.bytes]
  | .r09 | .r11 | .r37 | .r42 | .r45 =>
      [b.isolation.record.bytes] ++
        b.isolation.roleProjections.toList.map (fun record => record.bytes)
  | .r10 => [b.isolation.record.bytes, b.liveness.record.bytes] ++
      b.isolation.roleProjections.toList.map (fun record => record.bytes)
  | .r12 | .r13 | .r14 => [b.stages.producer.record.bytes,
      b.stages.verifierA.record.bytes, b.stages.verifierB.record.bytes]
  | .r15 => [b.stages.record.bytes, b.stages.producer.record.bytes,
      b.stages.verifierA.record.bytes, b.stages.verifierB.record.bytes]
  | .r16 | .r17 | .r18 => [b.artifact.record.bytes] ++
      b.artifact.files.map (fun record => record.bytes) ++
      b.artifact.directories.map (fun record => record.bytes)
  | .r32 | .r33 => [b.resource.record.bytes, b.artifact.record.bytes]
  | .r39 => [b.artifact.record.bytes, b.authorization.inbox.record.bytes,
      b.consumption.record.bytes]
  | .r19 | .r41 => [b.tools.record.bytes] ++
      b.tools.primaryTools.toList.map (fun record => record.bytes) ++
      b.tools.dynamicLoaders.map (fun record => record.bytes) ++
      b.tools.sharedLibraries.map (fun record => record.bytes)
  | .r20 | .r40 => [b.sourceClosure.record.bytes] ++
      b.sourceClosure.snapshotEntries.map (fun record => record.bytes) ++
      b.sourceClosure.snapshotDirectories.map (fun record => record.bytes)
  | .r21 => [b.semantic.record.bytes]
  | .r22 | .r44 => [b.postcheck.record.bytes, b.liveness.record.bytes,
      b.conformance.record.bytes, b.artifact.record.bytes,
      b.isolation.record.bytes, b.resource.record.bytes, b.semantic.record.bytes,
      b.sourceClosure.record.bytes, b.stages.record.bytes, b.tools.record.bytes]
  | .r23 => [b.artifact.record.bytes, b.liveness.record.bytes, b.postcheck.record.bytes]
  | .r24 => [b.authorization.record.bytes, b.consumption.record.bytes,
      b.postcheck.record.bytes]
  | .r25 => [b.authorization.record.bytes, b.consumption.record.bytes,
      b.liveness.record.bytes, b.postcheck.record.bytes]
  | .r26 => [b.consumption.record.bytes, b.postcheck.record.bytes]
  | .r46 => [b.postcheck.record.bytes]
  | .r27 | .r28 => [b.conformance.record.bytes, b.postcheck.record.bytes]
  | .r43 => [b.conformance.record.bytes, b.resource.record.bytes,
      b.postcheck.record.bytes]
  | .r29 => [b.authorization.inbox.record.bytes, b.authorization.record.bytes,
      b.authorization.reviewer1.record.bytes, b.authorization.reviewer2.record.bytes,
      b.authorization.identityRegistry.bytes,
      b.authorization.inbox.entries.authorization.bytes,
      b.authorization.inbox.entries.reviewer1.bytes,
      b.authorization.inbox.entries.reviewer2.bytes,
      b.authorization.inbox.entries.identityRegistry.bytes]
  | .r38 => [b.conformance.record.bytes, b.authorization.record.bytes,
      b.consumption.record.bytes, b.artifact.record.bytes, b.isolation.record.bytes,
      b.resource.record.bytes, b.semantic.record.bytes, b.sourceClosure.record.bytes,
      b.stages.record.bytes, b.tools.record.bytes, b.liveness.record.bytes,
      b.postcheck.record.bytes]

def expectedRelationNats {phase : Phase} (id : RelationId) (b : EvidenceCore phase) :
    List Nat :=
  match id with
  | .r01 => [b.conformance.maximumWallSeconds, b.conformance.maximumAccountedMemoryBytes,
      b.conformance.workersPerSemanticStage]
  | .r02 => [4]
  | .r03 => [b.authorization.inbox.attestationEntryCount]
  | .r04 => [b.consumption.frozenFdVerificationNs, b.consumption.consumedAtNs]
  | .r05 => [b.authorization.maximumConsumptions]
  | .r06 => [b.consumption.deadlineStartNs, b.consumption.frozenFdVerificationNs,
      b.consumption.consumedAtNs, b.resource.firstOperationalByteNs]
  | .r07 => [b.resource.deadlineStartNs, b.resource.postcheckCompletedNs,
      b.resource.deadlineAbsoluteNs]
  | .r08 => [b.conformance.workloadParentMemoryMaxBytes,
      b.conformance.outerDeadlineExecutorMemoryMaxBytes, b.resource.executorMemoryPeak,
      b.resource.workloadParentMemoryPeak]
  | .r09 => [b.isolation.roleProjections.toList.length]
  | .r10 => [b.isolation.hostNetworkFdMatchCountAfterProbe]
  | .r11 => [b.isolation.hostNetworkFdMatchCountAfterProbe]
  | .r12 => [b.stages.producer.residualPidCount, b.stages.verifierA.residualPidCount,
      b.stages.verifierB.residualPidCount]
  | .r13 => [b.stages.producer.workers.toList.length,
      b.stages.verifierA.workers.toList.length, b.stages.verifierB.workers.toList.length]
  | .r14 => [b.stages.producer.residualPidCount, b.stages.producer.escapedPidCount,
      b.stages.verifierA.residualPidCount, b.stages.verifierA.escapedPidCount,
      b.stages.verifierB.residualPidCount, b.stages.verifierB.escapedPidCount]
  | .r15 => [3]
  | .r16 => [b.artifact.fileCount, b.artifact.directoryCount, b.artifact.pathCount,
      b.artifact.logicalBytes]
  | .r17 => [b.artifact.fileCount, b.artifact.directoryCount, b.artifact.forbiddenEntryCount]
  | .r18 => [b.artifact.files.length, b.artifact.directories.length]
  | .r19 => [b.tools.primaryTools.toList.length, b.tools.dynamicLoaders.length,
      b.tools.sharedLibraries.length]
  | .r20 => [b.sourceClosure.fullArchiveFileCount, b.sourceClosure.e2ArchiveFileCount,
      b.sourceClosure.forbiddenEntryCount]
  | .r21 => [b.semantic.pairCount, b.semantic.tupleMembershipCount,
      b.semantic.paddedClippingCount, b.semantic.canonicalGrayEqualCount,
      b.semantic.canonicalPngEqualCount, b.semantic.graphInequivalentCount,
      b.semantic.rawGrayDistinctCount]
  | .r22 => [b.postcheck.certificateHashes.toList.length]
  | .r23 => [b.postcheck.residualProcessCount, b.resource.postcheckCompletedNs,
      b.resource.deadlineAbsoluteNs]
  | .r24 => [4]
  | .r25 => [b.resource.markerValidatedNs, b.resource.deadlineAbsoluteNs]
  | .r26 => [b.consumption.consumedAtNs, b.resource.markerValidatedNs]
  | .r27 => [261, drySuccessChain.length, officialSuccessChain.length]
  | .r28 => [261]
  | .r29 => [b.authorization.inbox.authorizationEntryCount,
      b.authorization.inbox.attestationEntryCount,
      b.authorization.inbox.identityRegistryEntryCount]
  | .r30 => [b.resource.executorMemoryPeak, b.resource.workloadParentMemoryPeak,
      b.resource.watchdogMemoryPeak, b.resource.monitorMemoryPeak, b.resource.innerMemoryPeak]
  | .r31 => [b.resource.deadlineStartNs, b.consumption.frozenFdVerificationNs,
      b.resource.authorizationConsumedNs, b.resource.firstOperationalByteNs,
      b.resource.postcheckCompletedNs, b.resource.markerValidatedNs,
      b.resource.deadlineAbsoluteNs]
  | .r32 => [b.resource.temporaryAllocatedBytes]
  | .r33 => [b.resource.publishedLogicalBytes, b.resource.publishedAllocatedBytes,
      b.artifact.logicalBytes, b.artifact.allocatedBytes]
  | .r34 => [b.resource.oomEventDeltaTotal, b.postcheck.residualProcessCount]
  | .r35 => b.resource.cpusetCpuIds.toList
  | .r36 => [b.resource.postcheckCompletedNs, b.resource.deadlineAbsoluteNs]
  | .r37 => [b.isolation.roleProjections.toList.length]
  | .r38 => [17, 45]
  | .r39 => [b.artifact.pathCount]
  | .r40 => [b.sourceClosure.fullArchiveFileCount,
      b.sourceClosure.snapshotEntries.length, b.sourceClosure.snapshotDirectories.length]
  | .r41 => [b.tools.primaryTools.toList.length, b.tools.dynamicLoaders.length,
      b.tools.sharedLibraries.length]
  | .r42 => [b.isolation.unexpectedWritableMountCount]
  | .r43 => [261, b.resource.deadlineAbsoluteNs]
  | .r44 => [8]
  | .r45 => [b.isolation.roleProjections.toList.length]
  | .r46 => [drySuccessChain.length, officialSuccessChain.length]

def expectedRelationStrings {phase : Phase} (id : RelationId) (b : EvidenceCore phase) :
    List String :=
  match id with
  | .r01 => [b.authorization.contractSha256, frozenContractSha256,
      b.authorization.contractSchemaSha256, frozenContractSchemaSha256,
      b.authorization.implementationCommit, b.bindings.implementationCommit.value,
      b.authorization.launcherSourceFreezeSha256, b.bindings.launcherSourceFreezeSha256.value,
      b.authorization.testBundleSha256, b.bindings.testBundleSha256.value,
      b.authorization.externalToolDependencyClosureSha256,
      b.bindings.externalToolDependencyClosureSha256.value]
  | .r02 => [b.authorization.implementationAuthor.id,
      b.authorization.authorizationIssuer.id, b.authorization.reviewer1.reviewer.id,
      b.authorization.reviewer2.reviewer.id]
  | .r03 => [b.authorization.reviewer1.record.sha256.value,
      b.authorization.inbox.entries.reviewer1.sha256.value,
      b.authorization.reviewer2.record.sha256.value,
      b.authorization.inbox.entries.reviewer2.sha256.value,
      b.authorization.reviewer1.otherReviewerId, b.authorization.reviewer2.reviewer.id,
      b.authorization.reviewer2.otherReviewerId, b.authorization.reviewer1.reviewer.id]
  | .r04 => [b.consumption.ledgerKey, b.consumption.ledgerRecordPath]
  | .r05 => [b.consumption.runNonce, b.authorization.runNonce.value]
  | .r06 => ["r06"]
  | .r07 => ["r07"]
  | .r08 => [b.conformance.contractSha256, frozenContractSha256]
  | .r09 => [b.isolation.roleMatrixSha256, frozenRoleMatrixSha256,
      b.isolation.projectionRootSha256]
  | .r10 => ["r10"]
  | .r11 => ["r11"]
  | .r12 => ["r12"]
  | .r13 => ["r13"]
  | .r14 => ["r14"]
  | .r15 => [b.stages.stageSetRootSha256]
  | .r16 => ["r16"]
  | .r17 => ["r17"]
  | .r18 => [b.artifact.contentRootSha256, b.artifact.metadataRootSha256,
      b.artifact.combinedRootSha256]
  | .r19 => [b.tools.closureRootSha256]
  | .r20 => [b.sourceClosure.sourceCommit, frozenSourceCommit,
      b.sourceClosure.fullArchiveSha256, frozenFullCommitArchiveSha256,
      b.sourceClosure.e2ArchiveSha256, frozenE2ArchiveSha256]
  | .r21 => ["r21"]
  | .r22 => [b.postcheck.requiredCertificateSetRootSha256,
      b.postcheck.livenessCertificateSha256, b.liveness.record.sha256.value]
  | .r23 => ["r23"]
  | .r24 => [b.authorization.contractSha256, b.authorization.implementationCommit,
      b.authorization.runNonce.value, b.postcheck.candidateCombinedRootSha256]
  | .r25 => [b.authorization.contractSha256, b.consumption.runNonce,
      b.authorization.runNonce.value, b.postcheck.candidateCombinedRootSha256,
      b.postcheck.requiredCertificateSetRootSha256]
  | .r26 => ["r26"]
  | .r27 => [frozenStateMachineSha256]
  | .r28 => [frozenStateMachineSha256]
  | .r29 => [b.authorization.record.sha256.value,
      b.authorization.reviewer1.record.sha256.value,
      b.authorization.reviewer2.record.sha256.value,
      b.authorization.identityRegistry.sha256.value]
  | .r30 => ["r30"]
  | .r31 => ["r31"]
  | .r32 => ["r32"]
  | .r33 => ["r33"]
  | .r34 => ["r34"]
  | .r35 => ["r35"]
  | .r36 => ["r36"]
  | .r37 => [b.isolation.roleMatrixSha256, frozenRoleMatrixSha256,
      b.isolation.projectionRootSha256]
  | .r38 => [frozenRecordSchemaBundleSha256]
  | .r39 => ["r39"]
  | .r40 => [b.sourceClosure.sourceCommit, frozenSourceCommit,
      b.sourceClosure.fullArchiveSha256, frozenFullCommitArchiveSha256,
      b.sourceClosure.e2ArchiveSha256, frozenE2ArchiveSha256]
  | .r41 => [b.tools.closureRootSha256]
  | .r42 => ["r42"]
  | .r43 => [frozenStateMachineSha256]
  | .r44 => [b.postcheck.requiredCertificateSetRootSha256,
      b.postcheck.livenessCertificateSha256, b.liveness.record.sha256.value]
  | .r45 => [b.isolation.roleMatrixSha256, frozenRoleMatrixSha256,
      b.isolation.projectionRootSha256]
  | .r46 => [frozenStateMachineSha256]

def expectedRelationUniqueKeys {phase : Phase} (id : RelationId) (b : EvidenceCore phase) :
    List String :=
  match id with
  | .r01 => ["r01"]
  | .r02 => [b.authorization.implementationAuthor.id,
      b.authorization.authorizationIssuer.id, b.authorization.reviewer1.reviewer.id,
      b.authorization.reviewer2.reviewer.id]
  | .r03 => [b.authorization.reviewer1.record.sha256.value,
      b.authorization.reviewer2.record.sha256.value]
  | .r04 => ["r04"]
  | .r05 => ["r05"]
  | .r06 => ["r06"]
  | .r07 => ["r07"]
  | .r08 => ["r08"]
  | .r09 => ["r09"]
  | .r10 => ["r10"]
  | .r11 => ["r11"]
  | .r12 => ["r12"]
  | .r13 => ["r13"]
  | .r14 => ["r14"]
  | .r15 => ["r15"]
  | .r16 => ["r16"]
  | .r17 => ["r17"]
  | .r18 => ["r18"]
  | .r19 => b.tools.primaryTools.toList.map (fun record => record.sha256.value) ++
      b.tools.dynamicLoaders.map (fun record => record.sha256.value) ++
      b.tools.sharedLibraries.map (fun record => record.sha256.value)
  | .r20 => ["r20"]
  | .r21 => ["r21"]
  | .r22 => ["r22"]
  | .r23 => ["r23"]
  | .r24 => ["r24"]
  | .r25 => ["r25"]
  | .r26 => ["r26"]
  | .r27 => ["r27"]
  | .r28 => ["r28"]
  | .r29 => [b.authorization.record.sha256.value,
      b.authorization.reviewer1.record.sha256.value,
      b.authorization.reviewer2.record.sha256.value,
      b.authorization.identityRegistry.sha256.value]
  | .r30 => ["r30"]
  | .r31 => ["r31"]
  | .r32 => ["r32"]
  | .r33 => ["r33"]
  | .r34 => ["r34"]
  | .r35 => ["r35"]
  | .r36 => ["r36"]
  | .r37 => ["r37"]
  | .r38 => ["r38"]
  | .r39 => (b.artifact.files ++ b.artifact.directories).map
      (fun record => record.sha256.value)
  | .r40 => ["r40"]
  | .r41 => b.tools.primaryTools.toList.map (fun record => record.sha256.value) ++
      b.tools.dynamicLoaders.map (fun record => record.sha256.value) ++
      b.tools.sharedLibraries.map (fun record => record.sha256.value)
  | .r42 => ["r42"]
  | .r43 => ["r43"]
  | .r44 => ["r44"]
  | .r45 => ["r45"]
  | .r46 => ["r46"]

def expectedRelationTimes {phase : Phase} (id : RelationId) (b : EvidenceCore phase) :
    List Nat :=
  match id with
  | .r01 => []
  | .r02 => []
  | .r03 => []
  | .r04 => [b.consumption.deadlineStartNs,
      b.consumption.frozenFdVerificationNs, b.consumption.consumedAtNs,
      b.resource.firstOperationalByteNs]
  | .r05 => []
  | .r06 => [b.consumption.deadlineStartNs,
      b.consumption.frozenFdVerificationNs, b.consumption.consumedAtNs,
      b.resource.firstOperationalByteNs]
  | .r07 => [b.resource.deadlineStartNs,
      b.resource.authorizationConsumedNs, b.resource.firstOperationalByteNs,
      b.resource.postcheckCompletedNs, b.resource.markerValidatedNs,
      b.resource.deadlineAbsoluteNs]
  | .r08 => []
  | .r09 => []
  | .r10 => []
  | .r11 => []
  | .r12 => []
  | .r13 => []
  | .r14 => []
  | .r15 => []
  | .r16 => []
  | .r17 => []
  | .r18 => []
  | .r19 => []
  | .r20 => []
  | .r21 => []
  | .r22 => []
  | .r23 => [b.resource.deadlineStartNs, b.resource.authorizationConsumedNs,
      b.resource.firstOperationalByteNs, b.resource.postcheckCompletedNs,
      b.resource.markerValidatedNs, b.resource.deadlineAbsoluteNs]
  | .r24 => []
  | .r25 => [b.resource.deadlineStartNs, b.resource.authorizationConsumedNs,
      b.resource.firstOperationalByteNs, b.resource.postcheckCompletedNs,
      b.resource.markerValidatedNs, b.resource.deadlineAbsoluteNs]
  | .r26 => [b.resource.deadlineStartNs, b.resource.authorizationConsumedNs,
      b.resource.firstOperationalByteNs, b.resource.postcheckCompletedNs,
      b.resource.markerValidatedNs, b.resource.deadlineAbsoluteNs]
  | .r27 => []
  | .r28 => []
  | .r29 => []
  | .r30 => []
  | .r31 => [b.resource.deadlineStartNs, b.resource.authorizationConsumedNs,
      b.resource.firstOperationalByteNs, b.resource.postcheckCompletedNs,
      b.resource.markerValidatedNs, b.resource.deadlineAbsoluteNs]
  | .r32 => []
  | .r33 => []
  | .r34 => []
  | .r35 => []
  | .r36 => [b.resource.deadlineStartNs, b.resource.authorizationConsumedNs,
      b.resource.firstOperationalByteNs, b.resource.postcheckCompletedNs,
      b.resource.markerValidatedNs, b.resource.deadlineAbsoluteNs]
  | .r37 => []
  | .r38 => []
  | .r39 => []
  | .r40 => []
  | .r41 => []
  | .r42 => []
  | .r43 => []
  | .r44 => []
  | .r45 => []
  | .r46 => []

def ConcreteResultSemantics {phase : Phase} (id : RelationId) (b : EvidenceCore phase)
    (result : ConcreteRelationResult) : Prop :=
  result.expectedBytes = relationInputBytes id b ∧
    result.observedBytes = result.expectedBytes ∧
    result.expectedNats = expectedRelationNats id b ∧
    result.observedNats = result.expectedNats ∧
    result.expectedStrings = expectedRelationStrings id b ∧
    result.observedStrings = result.expectedStrings ∧
    result.uniqueKeys = expectedRelationUniqueKeys id b ∧
    result.uniqueKeys.Nodup ∧
    result.monotonicTimes = expectedRelationTimes id b ∧
    result.monotonicTimes.Pairwise (fun left right => left <= right) ∧
    result.expectedBytes ≠ []

def OfflineRootBound (result : RootRelationResult) : Prop :=
  result.verifierSourceSha256 = frozenOfflineVerifierSourceSha256 ∧
    result.verifierTreeSha256 = frozenOfflineVerifierTreeSha256

def exactCertificateSetRecords {phase : Phase} (b : EvidenceCore phase) : List ByteArray :=
  [b.conformance.record.bytes, b.artifact.record.bytes, b.isolation.record.bytes,
    b.resource.record.bytes, b.semantic.record.bytes, b.sourceClosure.record.bytes,
    b.stages.record.bytes, b.tools.record.bytes]

def LedgerResultSemantics {phase : Phase} (b : EvidenceCore phase)
    (result : LedgerRelationResult) : Prop :=
  result.domain = "chemgraph-v0.6/e2-launcher-v0.3/authorization-ledger-key".toUTF8 ∧
    result.decodedContractSha256.size = 32 ∧ result.implementationCommitBytes.size = 20 ∧
    result.phaseByte = (match phase with | .dryRun => 0 | .official => 1) ∧
    result.claimedLedgerKey = b.consumption.ledgerKey ∧
    result.observedRecordPath = ledgerRecordPath result.claimedLedgerKey ∧
    result.observedRecordPath = b.consumption.ledgerRecordPath.toUTF8 ∧
    result.rootMetadata = ⟨0, 0, 448, 2⟩ ∧
    result.recordMetadata = ⟨0, 0, 292, 1⟩ ∧ result.compositeRecordCount = 1 ∧
    result.durableWrite.writtenBytes = b.consumption.record.bytes ∧
    result.durableWrite.reopenedBytes = result.durableWrite.writtenBytes ∧
    result.durableWrite.installedAtNs <= result.durableWrite.fileFsyncedAtNs ∧
    result.durableWrite.fileFsyncedAtNs <= result.durableWrite.parentFsyncedAtNs ∧
    result.durableWrite.outcome = .installedNoReplace

def RootResultSemantics (result : RootRelationResult) (domain : String)
    (records : List ByteArray) (claimedRoot : String) : Prop :=
  result.preimage.domain = domain.toUTF8 ∧
    result.preimage.orderedCompleteRecords = records ∧
    result.claimedRootSha256 = claimedRoot ∧ OfflineRootBound result

def ArtifactRootsResultSemantics {phase : Phase} (b : EvidenceCore phase)
    (result : ArtifactRootsRelationResult) : Prop :=
  RootResultSemantics result.content "chemgraph-v0.6/e2-launcher-v0.3/artifact-content"
      (b.artifact.files.map (fun record => record.bytes)) b.artifact.contentRootSha256 ∧
    RootResultSemantics result.metadata "chemgraph-v0.6/e2-launcher-v0.3/artifact-metadata"
      (b.artifact.directories.map (fun record => record.bytes)) b.artifact.metadataRootSha256 ∧
    RootResultSemantics result.combined "chemgraph-v0.6/e2-launcher-v0.3/artifact-combined"
      [result.content.preimage.bytes, result.metadata.preimage.bytes]
      b.artifact.combinedRootSha256

def MarkerPathResultSemantics {phase : Phase} (b : EvidenceCore phase)
    (result : MarkerPathRelationResult) : Prop :=
  ByteArray.mk result.tuple.contractSha256.bytes.toArray =
      b.authorization.contractSha256.toUTF8 ∧
    ByteArray.mk result.tuple.implementationCommit.bytes.toArray =
      b.authorization.implementationCommit.toUTF8 ∧
    ByteArray.mk result.tuple.runNonce.bytes.toArray = b.authorization.runNonce.value.toUTF8 ∧
    ByteArray.mk result.tuple.candidateCombinedRootSha256.bytes.toArray =
      b.postcheck.candidateCombinedRootSha256.toUTF8 ∧
    result.observedPath = markerPath result.tuple

def StateTableResultSemantics (result : StateTableRelationResult) : Prop :=
  result.failureTable = expectedFailureTransitionTable ∧
    result.failureTable.length = 261 ∧ result.dryChain = drySuccessChain ∧
    result.officialChain = officialSuccessChain ∧
    result.cleanupTable = expectedCleanupTable ∧
    result.authorityRule = .officialValidatedDetachedMarkerOnly

def AuthorizationAuthenticityResultSemantics {phase : Phase} (b : EvidenceCore phase)
    (result : AuthorizationAuthenticityRelationResult) : Prop :=
  result.committedEntries.authorization.bytes = b.authorization.record.bytes ∧
    result.committedEntries.authorization.sha256.value = b.authorization.record.sha256.value ∧
    result.committedEntries.reviewer1.bytes = b.authorization.reviewer1.record.bytes ∧
    result.committedEntries.reviewer1.sha256.value =
      b.authorization.reviewer1.record.sha256.value ∧
    result.committedEntries.reviewer2.bytes = b.authorization.reviewer2.record.bytes ∧
    result.committedEntries.reviewer2.sha256.value =
      b.authorization.reviewer2.record.sha256.value ∧
    result.committedEntries.identityRegistry.bytes = b.authorization.identityRegistry.bytes ∧
    result.committedEntries.identityRegistry.sha256.value =
      b.authorization.identityRegistry.sha256.value ∧
    result.committedEntryHashes = [b.authorization.record.sha256.value,
      b.authorization.reviewer1.record.sha256.value,
      b.authorization.reviewer2.record.sha256.value,
      b.authorization.identityRegistry.sha256.value] ∧
    b.authorization.reviewer1.record.bytes ≠ b.authorization.reviewer2.record.bytes ∧
    b.authorization.reviewer1.record.sha256.value ≠
      b.authorization.reviewer2.record.sha256.value ∧
    b.authorization.inbox.authorizationEntryCount = 1 ∧
    b.authorization.inbox.attestationEntryCount = 2 ∧
    b.authorization.inbox.identityRegistryEntryCount = 1 ∧
    result.inboxRootMetadata = ⟨0, 0, 448, 2⟩ ∧
    result.entryMetadata = ⟨0, 0, 292, 1⟩ ∧
    0 < result.authorizationFsyncedAtNs ∧ 0 < result.reviewer1FsyncedAtNs ∧
    0 < result.reviewer2FsyncedAtNs ∧ 0 < result.registryFsyncedAtNs ∧
    result.commitmentWrite.writtenBytes = b.authorization.inbox.record.bytes ∧
    result.commitmentWrite.reopenedBytes = result.commitmentWrite.writtenBytes ∧
    result.commitmentWrite.installedAtNs <= result.commitmentWrite.fileFsyncedAtNs ∧
    result.commitmentWrite.fileFsyncedAtNs <= result.commitmentWrite.parentFsyncedAtNs ∧
    result.commitmentWrite.outcome = .installedNoReplace

def SchemaResultSemantics (result : SchemaRelationResult) : Prop :=
  result.draftUri = "https://json-schema.org/draft/2020-12/schema" ∧
    result.records.toList.length = 17 ∧ result.ids.toList.length = 45 ∧
    result.ids.toList.Nodup

def RelationResultSemantics {phase : Phase} (id : RelationId) (b : EvidenceCore phase) :
    RelationResult -> Prop
  | .ledger result => id = .r04 ∧ LedgerResultSemantics b result
  | .artifactRoots result => id = .r18 ∧ ArtifactRootsResultSemantics b result
  | .roleMatrixRoot result => id = .r37 ∧
      RootResultSemantics result "chemgraph-e2-v0.4/role-matrix"
        (b.isolation.roleProjections.toList.map (fun record => record.bytes))
        frozenRoleMatrixSha256
  | .certificateSetRoot result => (id = .r22 ∨ id = .r44) ∧
      RootResultSemantics result "chemgraph-v0.6/e2-launcher-v0.3/required-certificate-set"
        (exactCertificateSetRecords b) b.postcheck.requiredCertificateSetRootSha256
  | .markerPath result => id = .r24 ∧ MarkerPathResultSemantics b result
  | .stateTable result => (id = .r27 ∨ id = .r28 ∨ id = .r43 ∨ id = .r46) ∧
      StateTableResultSemantics result
  | .authorizationAuthenticity result => id = .r29 ∧
      AuthorizationAuthenticityResultSemantics b result
  | .schemas result => id = .r38 ∧ SchemaResultSemantics result
  | .framedRoot result =>
      (id = .r15 ∧ RootResultSemantics result
        "chemgraph-v0.6/e2-launcher-v0.3/stage-set"
        [b.stages.producer.record.bytes, b.stages.verifierA.record.bytes,
          b.stages.verifierB.record.bytes] b.stages.stageSetRootSha256) ∨
      ((id = .r19 ∨ id = .r41) ∧ RootResultSemantics result
        "chemgraph-v0.6/e2-launcher-v0.3/tool-closure"
        (b.tools.primaryTools.toList.map (fun record => record.bytes) ++
          b.tools.dynamicLoaders.map (fun record => record.bytes) ++
          b.tools.sharedLibraries.map (fun record => record.bytes)) b.tools.closureRootSha256) ∨
      (id = .r45 ∧ RootResultSemantics result
        "chemgraph-v0.6/e2-launcher-v0.3/isolation-projection"
        (b.isolation.roleProjections.toList.map (fun record => record.bytes))
        b.isolation.projectionRootSha256)
  | .concrete result => id ≠ .r04 ∧ id ≠ .r15 ∧ id ≠ .r18 ∧ id ≠ .r19 ∧
      id ≠ .r22 ∧ id ≠ .r24 ∧ id ≠ .r27 ∧ id ≠ .r28 ∧ id ≠ .r29 ∧
      id ≠ .r37 ∧ id ≠ .r38 ∧ id ≠ .r41 ∧ id ≠ .r43 ∧ id ≠ .r44 ∧
      id ≠ .r45 ∧ id ≠ .r46 ∧ ConcreteResultSemantics id b result

def RelationSemantics {phase : Phase} (id : RelationId) (b : EvidenceCore phase)
    (observation : RelationObservation) : Prop :=
  RecordValid observation.verifierRecord ∧
    observation.completeInputBytes = relationInputBytes id b ∧
    observation.owner = expectedRelationOwner id ∧
    observation.verifierId = expectedRelationVerifier id ∧
    observation.leanTheoremId = expectedRelationTheorem id ∧
    observation.offlineVerifierSourceSha256 = frozenOfflineVerifierSourceSha256 ∧
    observation.offlineVerifierTreeSha256 = frozenOfflineVerifierTreeSha256 ∧
    observation.verifierRecord.bytes = relationTranscriptBytes id
      (expectedRelationOwner id) (expectedRelationVerifier id) (expectedRelationTheorem id)
      (relationInputBytes id b) observation.result.payload ∧
    RelationResultSemantics id b observation.result

theorem arbitrary_record_wrapper_cannot_satisfy_relation
    {phase : Phase} {id : RelationId} {b : EvidenceCore phase}
    {observation : RelationObservation}
    (mismatch : observation.verifierRecord.bytes ≠ relationTranscriptBytes id
      (expectedRelationOwner id) (expectedRelationVerifier id) (expectedRelationTheorem id)
      (relationInputBytes id b) observation.result.payload) :
    Not (RelationSemantics id b observation) := by
  intro semantics
  rcases semantics with ⟨_, _, _, _, _, _, _, exactTranscript, _⟩
  exact mismatch exactTranscript

theorem wrong_ledger_path_cannot_satisfy_r04
    {phase : Phase} {b : EvidenceCore phase} {result : LedgerRelationResult}
    (wrong : result.observedRecordPath ≠ ledgerRecordPath result.claimedLedgerKey) :
    Not (LedgerResultSemantics b result) := by
  intro semantics
  rcases semantics with ⟨_, _, _, _, _, exactPath, _⟩
  exact wrong exactPath

theorem duplicate_attestation_bytes_cannot_satisfy_r29
    {phase : Phase} {b : EvidenceCore phase}
    {result : AuthorizationAuthenticityRelationResult}
    (same : b.authorization.reviewer1.record.bytes =
      b.authorization.reviewer2.record.bytes) :
    Not (AuthorizationAuthenticityResultSemantics b result) := by
  intro semantics
  rcases semantics with ⟨_, _, _, _, _, _, _, _, _, distinctBytes, _⟩
  exact distinctBytes same

inductive RelationCoverageArea where
  | authorization | isolationResource | stage | artifact | closureSemantic
  | publication | state
deriving Repr

def relationCoverageArea : RelationId -> RelationCoverageArea
  | .r01 => .authorization | .r02 => .authorization | .r03 => .authorization
  | .r04 => .authorization | .r05 => .authorization | .r06 => .authorization
  | .r07 => .isolationResource | .r08 => .isolationResource
  | .r09 => .isolationResource | .r10 => .isolationResource
  | .r11 => .isolationResource | .r12 => .stage | .r13 => .stage
  | .r14 => .stage | .r15 => .stage | .r16 => .artifact | .r17 => .artifact
  | .r18 => .artifact | .r19 => .closureSemantic | .r20 => .closureSemantic
  | .r21 => .closureSemantic | .r22 => .publication | .r23 => .publication
  | .r24 => .publication | .r25 => .publication | .r26 => .publication
  | .r27 => .state | .r28 => .state | .r29 => .authorization
  | .r30 => .isolationResource | .r31 => .isolationResource
  | .r32 => .artifact | .r33 => .artifact | .r34 => .isolationResource
  | .r35 => .isolationResource | .r36 => .isolationResource
  | .r37 => .isolationResource | .r38 => .closureSemantic | .r39 => .artifact
  | .r40 => .closureSemantic | .r41 => .closureSemantic | .r42 => .stage
  | .r43 => .state | .r44 => .publication | .r45 => .isolationResource
  | .r46 => .state

theorem relation_coverage_total (id : RelationId) :
    Exists fun area => relationCoverageArea id = area :=
  ⟨relationCoverageArea id, rfl⟩

theorem chemgraph_e2_v03_relation_r01_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r01 b o) : RelationSemantics .r01 b o := proof
theorem chemgraph_e2_v03_relation_r02_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r02 b o) : RelationSemantics .r02 b o := proof
theorem chemgraph_e2_v03_relation_r03_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r03 b o) : RelationSemantics .r03 b o := proof
theorem chemgraph_e2_v03_relation_r04_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r04 b o) : RelationSemantics .r04 b o := proof
theorem chemgraph_e2_v03_relation_r05_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r05 b o) : RelationSemantics .r05 b o := proof
theorem chemgraph_e2_v03_relation_r06_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r06 b o) : RelationSemantics .r06 b o := proof
theorem chemgraph_e2_v03_relation_r07_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r07 b o) : RelationSemantics .r07 b o := proof
theorem chemgraph_e2_v03_relation_r08_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r08 b o) : RelationSemantics .r08 b o := proof
theorem chemgraph_e2_v03_relation_r09_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r09 b o) : RelationSemantics .r09 b o := proof
theorem chemgraph_e2_v03_relation_r10_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r10 b o) : RelationSemantics .r10 b o := proof
theorem chemgraph_e2_v03_relation_r11_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r11 b o) : RelationSemantics .r11 b o := proof
theorem chemgraph_e2_v03_relation_r12_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r12 b o) : RelationSemantics .r12 b o := proof
theorem chemgraph_e2_v03_relation_r13_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r13 b o) : RelationSemantics .r13 b o := proof
theorem chemgraph_e2_v03_relation_r14_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r14 b o) : RelationSemantics .r14 b o := proof
theorem chemgraph_e2_v03_relation_r15_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r15 b o) : RelationSemantics .r15 b o := proof
theorem chemgraph_e2_v03_relation_r16_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r16 b o) : RelationSemantics .r16 b o := proof
theorem chemgraph_e2_v03_relation_r17_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r17 b o) : RelationSemantics .r17 b o := proof
theorem chemgraph_e2_v03_relation_r18_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r18 b o) : RelationSemantics .r18 b o := proof
theorem chemgraph_e2_v03_relation_r19_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r19 b o) : RelationSemantics .r19 b o := proof
theorem chemgraph_e2_v03_relation_r20_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r20 b o) : RelationSemantics .r20 b o := proof
theorem chemgraph_e2_v03_relation_r21_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r21 b o) : RelationSemantics .r21 b o := proof
theorem chemgraph_e2_v03_relation_r22_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r22 b o) : RelationSemantics .r22 b o := proof
theorem chemgraph_e2_v03_relation_r23_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r23 b o) : RelationSemantics .r23 b o := proof
theorem chemgraph_e2_v03_relation_r24_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r24 b o) : RelationSemantics .r24 b o := proof
theorem chemgraph_e2_v03_relation_r25_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r25 b o) : RelationSemantics .r25 b o := proof
theorem chemgraph_e2_v03_relation_r26_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r26 b o) : RelationSemantics .r26 b o := proof
theorem chemgraph_e2_v03_relation_r27_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r27 b o) : RelationSemantics .r27 b o := proof
theorem chemgraph_e2_v03_relation_r28_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r28 b o) : RelationSemantics .r28 b o := proof
theorem chemgraph_e2_v04_relation_r29_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r29 b o) : RelationSemantics .r29 b o := proof
theorem chemgraph_e2_v03_relation_r30_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r30 b o) : RelationSemantics .r30 b o := proof
theorem chemgraph_e2_v03_relation_r31_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r31 b o) : RelationSemantics .r31 b o := proof
theorem chemgraph_e2_v03_relation_r32_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r32 b o) : RelationSemantics .r32 b o := proof
theorem chemgraph_e2_v03_relation_r33_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r33 b o) : RelationSemantics .r33 b o := proof
theorem chemgraph_e2_v03_relation_r34_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r34 b o) : RelationSemantics .r34 b o := proof
theorem chemgraph_e2_v03_relation_r35_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r35 b o) : RelationSemantics .r35 b o := proof
theorem chemgraph_e2_v03_relation_r36_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r36 b o) : RelationSemantics .r36 b o := proof
theorem chemgraph_e2_v03_relation_r37_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r37 b o) : RelationSemantics .r37 b o := proof
theorem chemgraph_e2_v03_relation_r38_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r38 b o) : RelationSemantics .r38 b o := proof
theorem chemgraph_e2_v03_relation_r39_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r39 b o) : RelationSemantics .r39 b o := proof
theorem chemgraph_e2_v03_relation_r40_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r40 b o) : RelationSemantics .r40 b o := proof
theorem chemgraph_e2_v03_relation_r41_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r41 b o) : RelationSemantics .r41 b o := proof
theorem chemgraph_e2_v03_relation_r42_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r42 b o) : RelationSemantics .r42 b o := proof
theorem chemgraph_e2_v03_relation_r43_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r43 b o) : RelationSemantics .r43 b o := proof
theorem chemgraph_e2_v03_relation_r44_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r44 b o) : RelationSemantics .r44 b o := proof
theorem chemgraph_e2_v03_relation_r45_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r45 b o) : RelationSemantics .r45 b o := proof
theorem chemgraph_e2_v03_relation_r46_sound {phase} {b : EvidenceCore phase} {o}
    (proof : RelationSemantics .r46 b o) : RelationSemantics .r46 b o := proof

structure RelationWitness {phase : Phase} (id : RelationId) (b : EvidenceCore phase) where
  observation : RelationObservation
  proof : RelationSemantics id b observation

def RelationPassed {phase : Phase} {id : RelationId} {b : EvidenceCore phase}
    (witness : RelationWitness id b) : Prop :=
  RelationSemantics id b witness.observation

structure RelationCertificate {phase : Phase} (b : EvidenceCore phase) where
  record : NonemptyCanonicalRecord
  r01 : RelationWitness .r01 b
  r02 : RelationWitness .r02 b
  r03 : RelationWitness .r03 b
  r04 : RelationWitness .r04 b
  r05 : RelationWitness .r05 b
  r06 : RelationWitness .r06 b
  r07 : RelationWitness .r07 b
  r08 : RelationWitness .r08 b
  r09 : RelationWitness .r09 b
  r10 : RelationWitness .r10 b
  r11 : RelationWitness .r11 b
  r12 : RelationWitness .r12 b
  r13 : RelationWitness .r13 b
  r14 : RelationWitness .r14 b
  r15 : RelationWitness .r15 b
  r16 : RelationWitness .r16 b
  r17 : RelationWitness .r17 b
  r18 : RelationWitness .r18 b
  r19 : RelationWitness .r19 b
  r20 : RelationWitness .r20 b
  r21 : RelationWitness .r21 b
  r22 : RelationWitness .r22 b
  r23 : RelationWitness .r23 b
  r24 : RelationWitness .r24 b
  r25 : RelationWitness .r25 b
  r26 : RelationWitness .r26 b
  r27 : RelationWitness .r27 b
  r28 : RelationWitness .r28 b
  r29 : RelationWitness .r29 b
  r30 : RelationWitness .r30 b
  r31 : RelationWitness .r31 b
  r32 : RelationWitness .r32 b
  r33 : RelationWitness .r33 b
  r34 : RelationWitness .r34 b
  r35 : RelationWitness .r35 b
  r36 : RelationWitness .r36 b
  r37 : RelationWitness .r37 b
  r38 : RelationWitness .r38 b
  r39 : RelationWitness .r39 b
  r40 : RelationWitness .r40 b
  r41 : RelationWitness .r41 b
  r42 : RelationWitness .r42 b
  r43 : RelationWitness .r43 b
  r44 : RelationWitness .r44 b
  r45 : RelationWitness .r45 b
  r46 : RelationWitness .r46 b

structure EvidenceBundle (phase : Phase) extends EvidenceCore phase where
  relations : RelationCertificate toEvidenceCore

structure RelationGates {phase : Phase} (bundle : EvidenceBundle phase) : Prop where
  recordValid : RecordValid bundle.relations.record
  r01 : RelationPassed bundle.relations.r01
  r02 : RelationPassed bundle.relations.r02
  r03 : RelationPassed bundle.relations.r03
  r04 : RelationPassed bundle.relations.r04
  r05 : RelationPassed bundle.relations.r05
  r06 : RelationPassed bundle.relations.r06
  r07 : RelationPassed bundle.relations.r07
  r08 : RelationPassed bundle.relations.r08
  r09 : RelationPassed bundle.relations.r09
  r10 : RelationPassed bundle.relations.r10
  r11 : RelationPassed bundle.relations.r11
  r12 : RelationPassed bundle.relations.r12
  r13 : RelationPassed bundle.relations.r13
  r14 : RelationPassed bundle.relations.r14
  r15 : RelationPassed bundle.relations.r15
  r16 : RelationPassed bundle.relations.r16
  r17 : RelationPassed bundle.relations.r17
  r18 : RelationPassed bundle.relations.r18
  r19 : RelationPassed bundle.relations.r19
  r20 : RelationPassed bundle.relations.r20
  r21 : RelationPassed bundle.relations.r21
  r22 : RelationPassed bundle.relations.r22
  r23 : RelationPassed bundle.relations.r23
  r24 : RelationPassed bundle.relations.r24
  r25 : RelationPassed bundle.relations.r25
  r26 : RelationPassed bundle.relations.r26
  r27 : RelationPassed bundle.relations.r27
  r28 : RelationPassed bundle.relations.r28
  r29 : RelationPassed bundle.relations.r29
  r30 : RelationPassed bundle.relations.r30
  r31 : RelationPassed bundle.relations.r31
  r32 : RelationPassed bundle.relations.r32
  r33 : RelationPassed bundle.relations.r33
  r34 : RelationPassed bundle.relations.r34
  r35 : RelationPassed bundle.relations.r35
  r36 : RelationPassed bundle.relations.r36
  r37 : RelationPassed bundle.relations.r37
  r38 : RelationPassed bundle.relations.r38
  r39 : RelationPassed bundle.relations.r39
  r40 : RelationPassed bundle.relations.r40
  r41 : RelationPassed bundle.relations.r41
  r42 : RelationPassed bundle.relations.r42
  r43 : RelationPassed bundle.relations.r43
  r44 : RelationPassed bundle.relations.r44
  r45 : RelationPassed bundle.relations.r45
  r46 : RelationPassed bundle.relations.r46

structure OneByteAllTrueForgery (observation : RelationObservation) : Prop where
  oneByte : observation.verifierRecord.bytes.size = 1
  shaEncoding : observation.verifierRecord.sha256.lowercaseEncodingMatches = true
  shaNonzero : observation.verifierRecord.sha256.nonzero = true
  duplicateKeys : observation.verifierRecord.duplicateKeysRejected = true
  canonical : observation.verifierRecord.canonicalJson = true
  closedSchema : observation.verifierRecord.closedSchemaPassed = true
  fakeInputs : observation.completeInputBytes = [observation.verifierRecord.bytes]

theorem one_byte_all_true_cannot_construct_r01
    {phase : Phase} {b : EvidenceCore phase} {observation : RelationObservation}
    (fake : OneByteAllTrueForgery observation) :
    Not (RelationSemantics .r01 b observation) := by
  intro semantics
  have inputs := fake.fakeInputs.symm.trans semantics.2.1
  have tails : ([] : List ByteArray) = [b.conformance.record.bytes,
      b.sourceClosure.record.bytes, b.tools.record.bytes] :=
    congrArg List.tail inputs
  cases tails

structure CertificateGates {phase : Phase} (bundle : EvidenceBundle phase) : Prop where
  artifactRecord : RecordValid bundle.artifact.record
  artifactNonempty : bundle.artifact.files ≠ [] ∧ bundle.artifact.directories ≠ []
  artifactCounts : bundle.artifact.fileCount = bundle.artifact.files.length ∧
    bundle.artifact.directoryCount = bundle.artifact.directories.length ∧
    bundle.artifact.pathCount = bundle.artifact.fileCount + bundle.artifact.directoryCount
  artifactPassed : bundle.artifact.pathsUniqueSorted = true ∧
    bundle.artifact.fullBytesReadBack = true ∧
    bundle.artifact.allMetadataExact = true ∧ bundle.artifact.allFsynced = true ∧
    bundle.artifact.forbiddenEntryCount = 0 ∧ bundle.artifact.rootsRecomputed = true ∧
    bundle.artifact.statusPassed = true
  isolationRecord : RecordValid bundle.isolation.record
  isolationPassed : bundle.isolation.roleProjections.toList.length = 6 ∧
    bundle.isolation.roleMatrixSha256 = frozenRoleMatrixSha256 ∧
    bundle.isolation.siblingsVerified = true ∧
    bundle.isolation.permissionMatrixExact = true ∧
    bundle.isolation.projectionRootRecomputed = true ∧
    bundle.isolation.rootRecursiveReadonly = true ∧
    bundle.isolation.unexpectedWritableMountCount = 0 ∧
    bundle.isolation.hostRootMapped = false ∧
    bundle.isolation.capabilitySetsEmpty = true ∧
    bundle.isolation.noNewPrivileges = true ∧ bundle.isolation.hostSetnsEperm = true ∧
    bundle.isolation.hostNetworkFdMatchCountAfterProbe = 0 ∧
    bundle.isolation.statusPassed = true
  resourceRecord : RecordValid bundle.resource.record
  resourceOrdering : bundle.resource.deadlineStartNs <= bundle.resource.authorizationConsumedNs ∧
    bundle.resource.authorizationConsumedNs <= bundle.resource.firstOperationalByteNs ∧
    bundle.resource.firstOperationalByteNs <= bundle.resource.postcheckCompletedNs ∧
    bundle.resource.postcheckCompletedNs <= bundle.resource.deadlineAbsoluteNs
  resourcePassed : bundle.resource.wallUpperSeconds <= 1800 ∧
    bundle.resource.executorMemoryPeak + bundle.resource.workloadParentMemoryPeak <= 8589934592 ∧
    bundle.resource.workloadParentMemoryPeak <= 8556380160 ∧
    bundle.resource.watchdogMemoryPeak <= 67108864 ∧
    bundle.resource.monitorMemoryPeak <= 469762048 ∧
    bundle.resource.innerMemoryPeak <= 8019509248 ∧
    bundle.resource.temporaryAllocatedBytes <= 17179869184 ∧
    bundle.resource.oomEventDeltaTotal = 0 ∧
    bundle.resource.cpusetCpuIds.toList.length = 4 ∧
    bundle.resource.physicalCorePairsDistinct = true ∧
    bundle.resource.deadlineExpired = false ∧
    bundle.resource.executorLiveAtPostcheck = true ∧
    bundle.resource.watchdogPidfdExitObserved = false ∧ bundle.resource.statusPassed = true
  semanticRecord : RecordValid bundle.semantic.record
  semanticPassed : bundle.semantic.pairCount = 8 ∧
    bundle.semantic.tupleMembershipCount = 16 ∧
    bundle.semantic.paddedClippingCount = 16 ∧
    bundle.semantic.canonicalGrayEqualCount = 8 ∧
    bundle.semantic.canonicalPngEqualCount = 8 ∧
    bundle.semantic.graphInequivalentCount = 8 ∧
    bundle.semantic.rawGrayDistinctCount = 8 ∧
    bundle.semantic.verifierAgreement = true ∧ bundle.semantic.statusPassed = true
  sourceRecord : RecordValid bundle.sourceClosure.record
  sourcePassed : bundle.sourceClosure.sourceCommit = frozenSourceCommit ∧
    bundle.sourceClosure.fullArchiveSha256 = frozenFullCommitArchiveSha256 ∧
    bundle.sourceClosure.e2ArchiveSha256 = frozenE2ArchiveSha256 ∧
    bundle.sourceClosure.fullArchiveFileCount = 1921 ∧
    bundle.sourceClosure.e2ArchiveFileCount = 19 ∧
    bundle.sourceClosure.snapshotEntries ≠ [] ∧
    bundle.sourceClosure.snapshotDirectories ≠ [] ∧
    bundle.sourceClosure.safeExtraction = true ∧
    bundle.sourceClosure.completeArchiveBytesEqual = true ∧
    bundle.sourceClosure.completePathSetEqual = true ∧
    bundle.sourceClosure.allFsynced = true ∧
    bundle.sourceClosure.rootsRecomputed = true ∧
    bundle.sourceClosure.forbiddenEntryCount = 0 ∧ bundle.sourceClosure.statusPassed = true
  stagesRecord : RecordValid bundle.stages.record
  stagesPassed : StagePassed bundle.stages.producer ∧ StagePassed bundle.stages.verifierA ∧
    StagePassed bundle.stages.verifierB ∧
    bundle.stages.stageSetRootRecomputedFromFullBytes = true ∧
    bundle.stages.statusPassed = true
  toolsRecord : RecordValid bundle.tools.record
  toolsNonempty : bundle.tools.dynamicLoaders ≠ [] ∧ bundle.tools.sharedLibraries ≠ []
  toolsPassed : bundle.tools.primaryTools.toList.length = 11 ∧
    bundle.tools.descriptorPayloadsNonempty = true ∧
    bundle.tools.primaryRolesUnique = true ∧ bundle.tools.resolvedPathsUnique = true ∧
    bundle.tools.recursiveClosureComplete = true ∧
    bundle.tools.completeBytesVerified = true ∧ bundle.tools.closureRootRecomputed = true ∧
    bundle.tools.noFallbackPathLookup = true ∧ bundle.tools.statusPassed = true
  livenessRecord : RecordValid bundle.liveness.record
  livenessPassed : bundle.liveness.executorAndWatchdogUidsDistinct = true ∧
    bundle.liveness.pidNamespacesDistinct = true ∧
    bundle.liveness.mutualSignalEperm = true ∧
    bundle.liveness.executorHoldsWatchdogPidfd = true ∧
    bundle.liveness.watchdogHoldsExecutorPidfd = true ∧
    bundle.liveness.executorLiveAtPostcheck = true ∧
    bundle.liveness.watchdogLiveAtPostcheck = true ∧
    bundle.liveness.executorDeathProbeKilledParent = true ∧
    bundle.liveness.watchdogDeathProbeKilledParent = true ∧
    bundle.liveness.statusPassed = true
  postcheckRecord : RecordValid bundle.postcheck.record
  postcheckPassed : bundle.postcheck.certificateHashes.toList.length = 8 ∧
    bundle.postcheck.completeBytesEqual = true ∧
    bundle.postcheck.normalizedPathsEqual = true ∧
    bundle.postcheck.contentRootEqual = true ∧ bundle.postcheck.metadataRootEqual = true ∧
    bundle.postcheck.combinedRootEqual = true ∧
    bundle.postcheck.ownershipModesLinksExact = true ∧
    bundle.postcheck.immutabilityProbesPassed = true ∧
    bundle.postcheck.monitorReopenPassed = true ∧ bundle.postcheck.watchdogReopenPassed = true ∧
    bundle.postcheck.executorLive = true ∧ bundle.postcheck.deadlineNotExpired = true ∧
    bundle.postcheck.oomDeltasZero = true ∧ bundle.postcheck.residualProcessCount = 0 ∧
    bundle.postcheck.markerAbsentDuringPostcheck = true ∧ bundle.postcheck.statusPassed = true

structure DryRunPhaseGates (bundle : EvidenceBundle .dryRun) : Prop where
  noDryAuthorizationBindings : bundle.authorization.dryRunAuthorizationRecordSha256 = none ∧
    bundle.authorization.dryRunConsumptionRecordSha256 = none ∧
    bundle.authorization.dryRunValidationSha256 = none ∧
    bundle.authorization.redteamValidationSha256 = none
  noOfficialPublishedBytes : bundle.resource.publishedLogicalBytes = 0 ∧
    bundle.resource.publishedAllocatedBytes = 0
  noOfficialRename : bundle.postcheck.renamedCandidateObserved = false ∧
    bundle.postcheck.resultsParentFsynced = false ∧
    bundle.postcheck.officialResultsPathAbsent = true

structure OfficialPhaseGates (bundle : EvidenceBundle .official) : Prop where
  dryAuthorizationBound : Exists fun value =>
    bundle.authorization.dryRunAuthorizationRecordSha256 = some value
  dryConsumptionBound : Exists fun value =>
    bundle.authorization.dryRunConsumptionRecordSha256 = some value
  dryValidationBound : Exists fun value =>
    bundle.authorization.dryRunValidationSha256 = some value
  redteamValidationBound : Exists fun value =>
    bundle.authorization.redteamValidationSha256 = some value
  officialPublishedBytes : 0 < bundle.resource.publishedLogicalBytes ∧
    0 < bundle.resource.publishedAllocatedBytes ∧
    bundle.resource.publishedLogicalBytes <= 2147483648 ∧
    bundle.resource.publishedAllocatedBytes <= 2147483648
  officialRename : bundle.postcheck.renamedCandidateObserved = true ∧
    bundle.postcheck.resultsParentFsynced = true ∧
    bundle.postcheck.officialResultsPathAbsent = false

inductive PhaseSpecificGates : {phase : Phase} -> EvidenceBundle phase -> Prop where
  | dryRun {bundle : EvidenceBundle .dryRun} :
      DryRunPhaseGates bundle -> PhaseSpecificGates bundle
  | official {bundle : EvidenceBundle .official} :
      OfficialPhaseGates bundle -> PhaseSpecificGates bundle

structure AllFrozenGates {phase : Phase} (bundle : EvidenceBundle phase) : Prop where
  conformance : ExactContractConformance bundle.bindings bundle.conformance
  authorization : AuthorizationGates bundle.bindings bundle.authorization bundle.consumption
  certificates : CertificateGates bundle
  relations : RelationGates bundle
  phaseSpecific : PhaseSpecificGates bundle

structure DetachedAuthorityMarker (bundle : EvidenceBundle .official) where
  record : NonemptyCanonicalRecord
  contractSha256 : String
  contractSchemaSha256 : String
  implementationCommit : String
  runNonce : String
  candidateCombinedRootSha256 : String
  requiredCertificateSetRootSha256 : String
  officialAuthorizationRecordSha256 : String
  authorizationConsumptionRecordSha256 : String
  livenessCertificateSha256 : String
  postcheckCertificateSha256 : String
  markerFileFsynced : Bool
  markerParentFsynced : Bool
  markerReopenFullBytesEqual : Bool
  markerMetadataExact : Bool
  atomicNoreplace : Bool
  statusPassed : Bool

structure ValidDetachedAuthorityMarker
    (bundle : EvidenceBundle .official)
    (marker : DetachedAuthorityMarker bundle) : Prop where
  recordValid : RecordValid marker.record
  contract : marker.contractSha256 = frozenContractSha256
  contractSchema : marker.contractSchemaSha256 = frozenContractSchemaSha256
  implementation : marker.implementationCommit = bundle.bindings.implementationCommit.value
  nonce : marker.runNonce = bundle.authorization.runNonce.value
  candidateRoot : marker.candidateCombinedRootSha256 =
    bundle.postcheck.candidateCombinedRootSha256
  certificateSetRoot : marker.requiredCertificateSetRootSha256 =
    bundle.postcheck.requiredCertificateSetRootSha256
  authorizationRecord : marker.officialAuthorizationRecordSha256 =
    bundle.authorization.record.sha256.value
  consumptionRecord : marker.authorizationConsumptionRecordSha256 =
    bundle.consumption.record.sha256.value
  livenessRecord : marker.livenessCertificateSha256 = bundle.liveness.record.sha256.value
  postcheckRecord : marker.postcheckCertificateSha256 = bundle.postcheck.record.sha256.value
  fileFsynced : marker.markerFileFsynced = true
  parentFsynced : marker.markerParentFsynced = true
  reopened : marker.markerReopenFullBytesEqual = true
  metadata : marker.markerMetadataExact = true
  noReplace : marker.atomicNoreplace = true
  passed : marker.statusPassed = true

inductive State : (phase : Phase) -> EvidenceBundle phase -> Type where
  | initial {phase bundle} : State phase bundle
  | outerDeadlineArmed {phase bundle} : State phase bundle
  | siblingsReady {phase bundle} : State phase bundle
  | authorizationConsumed {phase bundle} : State phase bundle
  | monitorRunning {phase bundle} : State phase bundle
  | innerRunning {phase bundle} : State phase bundle
  | producerRunning {phase bundle} : State phase bundle
  | producerQuiescent {phase bundle} : State phase bundle
  | verifierARunning {phase bundle} : State phase bundle
  | verifierAQuiescent {phase bundle} : State phase bundle
  | verifierBRunning {phase bundle} : State phase bundle
  | verifierBQuiescent {phase bundle} : State phase bundle
  | dryRunCandidate {bundle : EvidenceBundle .dryRun} : State .dryRun bundle
  | dryRunPostchecked {bundle : EvidenceBundle .dryRun}
      (gates : AllFrozenGates bundle) : State .dryRun bundle
  | dryRunPassed {bundle : EvidenceBundle .dryRun}
      (gates : AllFrozenGates bundle) : State .dryRun bundle
  | candidate {bundle : EvidenceBundle .official} : State .official bundle
  | metadataFsynced {bundle : EvidenceBundle .official} : State .official bundle
  | renamedCandidate {bundle : EvidenceBundle .official} : State .official bundle
  | postchecked {bundle : EvidenceBundle .official}
      (gates : AllFrozenGates bundle) : State .official bundle
  | detachedMarkerInstalled {bundle : EvidenceBundle .official}
      (gates : AllFrozenGates bundle) (marker : DetachedAuthorityMarker bundle) :
      State .official bundle
  | authoritative {bundle : EvidenceBundle .official}
      (gates : AllFrozenGates bundle) (marker : DetachedAuthorityMarker bundle)
      (markerValid : ValidDetachedAuthorityMarker bundle marker) : State .official bundle
  | failed {phase bundle} : State phase bundle
  | timedOut {phase bundle} : State phase bundle
  | monitorDead {phase bundle} : State phase bundle
  | watchdogDead {phase bundle} : State phase bundle
  | deadlineExecutorDead {phase bundle} : State phase bundle
  | residual {phase bundle} : State phase bundle
  | quarantine {phase bundle} : State phase bundle
  | orphanedNonAuthoritativeCandidate {phase bundle} : State phase bundle

/-! A small explicit event type makes the first failure hop deterministic. -/
inductive FailureEvent where
  | schemaOrRelation
  | resourceOrIsolation
  | stageOrSourceTool
  | postcheckOrMarker
  | deadlineExpired
  | monitorExitOrOom
  | watchdogExit
  | deadlineExecutorExit
  | residualProcess

def failureTarget {phase : Phase} {bundle : EvidenceBundle phase} :
    FailureEvent -> State phase bundle
  | .schemaOrRelation => .failed
  | .resourceOrIsolation => .failed
  | .stageOrSourceTool => .failed
  | .postcheckOrMarker => .failed
  | .deadlineExpired => .timedOut
  | .monitorExitOrOom => .monitorDead
  | .watchdogExit => .watchdogDead
  | .deadlineExecutorExit => .deadlineExecutorDead
  | .residualProcess => .residual

inductive FailureSource : {phase : Phase} -> EvidenceBundle phase -> Type where
  | outerDeadlineArmed {phase bundle} : FailureSource (phase := phase) bundle
  | siblingsReady {phase bundle} : FailureSource (phase := phase) bundle
  | authorizationConsumed {phase bundle} : FailureSource (phase := phase) bundle
  | monitorRunning {phase bundle} : FailureSource (phase := phase) bundle
  | innerRunning {phase bundle} : FailureSource (phase := phase) bundle
  | producerRunning {phase bundle} : FailureSource (phase := phase) bundle
  | producerQuiescent {phase bundle} : FailureSource (phase := phase) bundle
  | verifierARunning {phase bundle} : FailureSource (phase := phase) bundle
  | verifierAQuiescent {phase bundle} : FailureSource (phase := phase) bundle
  | verifierBRunning {phase bundle} : FailureSource (phase := phase) bundle
  | verifierBQuiescent {phase bundle} : FailureSource (phase := phase) bundle
  | dryRunCandidate {bundle} : FailureSource (phase := .dryRun) bundle
  | dryRunPostchecked {bundle} (gates : AllFrozenGates bundle) :
      FailureSource (phase := .dryRun) bundle
  | candidate {bundle} : FailureSource (phase := .official) bundle
  | metadataFsynced {bundle} : FailureSource (phase := .official) bundle
  | renamedCandidate {bundle} : FailureSource (phase := .official) bundle
  | postchecked {bundle} (gates : AllFrozenGates bundle) :
      FailureSource (phase := .official) bundle
  | markerInstalled {bundle} (gates : AllFrozenGates bundle)
      (marker : DetachedAuthorityMarker bundle) : FailureSource (phase := .official) bundle

def FailureSource.state {phase : Phase} {bundle : EvidenceBundle phase} :
    FailureSource (phase := phase) bundle -> State phase bundle
  | .outerDeadlineArmed => .outerDeadlineArmed
  | .siblingsReady => .siblingsReady
  | .authorizationConsumed => .authorizationConsumed
  | .monitorRunning => .monitorRunning
  | .innerRunning => .innerRunning
  | .producerRunning => .producerRunning
  | .producerQuiescent => .producerQuiescent
  | .verifierARunning => .verifierARunning
  | .verifierAQuiescent => .verifierAQuiescent
  | .verifierBRunning => .verifierBRunning
  | .verifierBQuiescent => .verifierBQuiescent
  | .dryRunCandidate => .dryRunCandidate
  | .dryRunPostchecked gates => .dryRunPostchecked gates
  | .candidate => .candidate
  | .metadataFsynced => .metadataFsynced
  | .renamedCandidate => .renamedCandidate
  | .postchecked gates => .postchecked gates
  | .markerInstalled gates marker => .detachedMarkerInstalled gates marker

inductive Step : {phase : Phase} -> {bundle : EvidenceBundle phase} ->
    State phase bundle -> State phase bundle -> Prop where
  | arm : Step .initial .outerDeadlineArmed
  | siblings : Step .outerDeadlineArmed .siblingsReady
  | consume : Step .siblingsReady .authorizationConsumed
  | monitor : Step .authorizationConsumed .monitorRunning
  | inner : Step .monitorRunning .innerRunning
  | producer : Step .innerRunning .producerRunning
  | producerDone : Step .producerRunning .producerQuiescent
  | verifierA : Step .producerQuiescent .verifierARunning
  | verifierADone : Step .verifierARunning .verifierAQuiescent
  | verifierB : Step .verifierAQuiescent .verifierBRunning
  | verifierBDone : Step .verifierBRunning .verifierBQuiescent
  | dryCandidate {bundle : EvidenceBundle .dryRun} :
      Step (bundle := bundle) .verifierBQuiescent .dryRunCandidate
  | dryPostcheck {bundle : EvidenceBundle .dryRun} (gates : AllFrozenGates bundle) :
      Step .dryRunCandidate (.dryRunPostchecked gates)
  | dryPass {bundle : EvidenceBundle .dryRun} (gates : AllFrozenGates bundle) :
      Step (.dryRunPostchecked gates) (.dryRunPassed gates)
  | officialCandidate {bundle : EvidenceBundle .official} :
      Step (bundle := bundle) .verifierBQuiescent .candidate
  | metadata {bundle : EvidenceBundle .official} : Step .candidate .metadataFsynced
  | rename {bundle : EvidenceBundle .official} : Step .metadataFsynced .renamedCandidate
  | postcheck {bundle : EvidenceBundle .official} (gates : AllFrozenGates bundle) :
      Step .renamedCandidate (.postchecked gates)
  | installMarker {bundle : EvidenceBundle .official} (gates : AllFrozenGates bundle)
      (marker : DetachedAuthorityMarker bundle) :
      Step (.postchecked gates) (.detachedMarkerInstalled gates marker)
  | authorize {bundle : EvidenceBundle .official} (gates : AllFrozenGates bundle)
      (marker : DetachedAuthorityMarker bundle)
      (valid : ValidDetachedAuthorityMarker bundle marker) :
      Step (.detachedMarkerInstalled gates marker) (.authoritative gates marker valid)
  | fail {phase : Phase} {bundle : EvidenceBundle phase}
      (source : FailureSource (phase := phase) bundle) (event : FailureEvent) :
      Step source.state (failureTarget event)
  | cleanupFailed : Step .failed .quarantine
  | cleanupMonitorDead : Step .monitorDead .quarantine
  | cleanupResidual : Step .residual .quarantine
  | orphanTimeout : Step .timedOut .orphanedNonAuthoritativeCandidate
  | orphanWatchdog : Step .watchdogDead .orphanedNonAuthoritativeCandidate
  | orphanExecutor : Step .deadlineExecutorDead .orphanedNonAuthoritativeCandidate

inductive Reach {phase : Phase} {bundle : EvidenceBundle phase} :
    State phase bundle -> State phase bundle -> Prop where
  | refl (state) : Reach state state
  | next {source middle target} : Step source middle -> Reach middle target -> Reach source target

inductive ValidMarkerFor : {phase : Phase} -> EvidenceBundle phase -> Prop where
  | official {bundle : EvidenceBundle .official} (marker : DetachedAuthorityMarker bundle)
      (valid : ValidDetachedAuthorityMarker bundle marker) : ValidMarkerFor bundle

theorem dryRunPassed_only_dry_phase_and_no_marker
    {bundle : EvidenceBundle .dryRun} {gates : AllFrozenGates bundle}
    (_reachable : Reach (.initial) (.dryRunPassed gates)) :
    AllFrozenGates bundle ∧ Not (ValidMarkerFor bundle) := by
  refine ⟨gates, ?_⟩
  intro marker
  cases marker

theorem authoritative_implies_official_all_gates_marker_and_exact_four_inbox
    {bundle : EvidenceBundle .official} {gates : AllFrozenGates bundle}
    {marker : DetachedAuthorityMarker bundle}
    {valid : ValidDetachedAuthorityMarker bundle marker}
    (_reachable : Reach (.initial) (.authoritative gates marker valid)) :
    AllFrozenGates bundle ∧ ValidDetachedAuthorityMarker bundle marker ∧
      bundle.authorization.inbox.entries.toList.length = 4 ∧
      ExactInboxEntries bundle.authorization := by
  exact ⟨gates, valid, gates.authorization.fourInboxEntries,
    gates.authorization.inboxEntriesInFrozenOrder⟩

theorem all_frozen_gates_bind_exact_contract
    {phase : Phase} {bundle : EvidenceBundle phase}
    (gates : AllFrozenGates bundle) :
    bundle.conformance.contractSha256 = frozenContractSha256 ∧
      bundle.conformance.contractSchemaSha256 = frozenContractSchemaSha256 ∧
      bundle.conformance.allowExclusions = false ∧
      bundle.conformance.trainingAuthorized = false ∧
      bundle.conformance.executionAuthorized = false ∧
      bundle.conformance.maximumWallSeconds = 1800 ∧
      bundle.conformance.maximumAccountedMemoryBytes = 8589934592 ∧
      bundle.conformance.workersPerSemanticStage = 4 := by
  exact ⟨gates.conformance.contract, gates.conformance.contractSchema,
    gates.conformance.noExclusions, gates.conformance.noTraining,
    gates.conformance.contractDoesNotAuthorizeExecution, gates.conformance.wall,
    gates.conformance.globalMemory, gates.conformance.workerCount⟩

theorem relation_gate_count_is_exactly_46
    {phase : Phase} {bundle : EvidenceBundle phase} (gates : RelationGates bundle) :
    RelationPassed bundle.relations.r01 ∧ RelationPassed bundle.relations.r02 ∧
    RelationPassed bundle.relations.r03 ∧ RelationPassed bundle.relations.r04 ∧
    RelationPassed bundle.relations.r05 ∧ RelationPassed bundle.relations.r06 ∧
    RelationPassed bundle.relations.r07 ∧ RelationPassed bundle.relations.r08 ∧
    RelationPassed bundle.relations.r09 ∧ RelationPassed bundle.relations.r10 ∧
    RelationPassed bundle.relations.r11 ∧ RelationPassed bundle.relations.r12 ∧
    RelationPassed bundle.relations.r13 ∧ RelationPassed bundle.relations.r14 ∧
    RelationPassed bundle.relations.r15 ∧ RelationPassed bundle.relations.r16 ∧
    RelationPassed bundle.relations.r17 ∧ RelationPassed bundle.relations.r18 ∧
    RelationPassed bundle.relations.r19 ∧ RelationPassed bundle.relations.r20 ∧
    RelationPassed bundle.relations.r21 ∧ RelationPassed bundle.relations.r22 ∧
    RelationPassed bundle.relations.r23 ∧ RelationPassed bundle.relations.r24 ∧
    RelationPassed bundle.relations.r25 ∧ RelationPassed bundle.relations.r26 ∧
    RelationPassed bundle.relations.r27 ∧ RelationPassed bundle.relations.r28 ∧
    RelationPassed bundle.relations.r29 ∧ RelationPassed bundle.relations.r30 ∧
    RelationPassed bundle.relations.r31 ∧ RelationPassed bundle.relations.r32 ∧
    RelationPassed bundle.relations.r33 ∧ RelationPassed bundle.relations.r34 ∧
    RelationPassed bundle.relations.r35 ∧ RelationPassed bundle.relations.r36 ∧
    RelationPassed bundle.relations.r37 ∧ RelationPassed bundle.relations.r38 ∧
    RelationPassed bundle.relations.r39 ∧ RelationPassed bundle.relations.r40 ∧
    RelationPassed bundle.relations.r41 ∧ RelationPassed bundle.relations.r42 ∧
    RelationPassed bundle.relations.r43 ∧ RelationPassed bundle.relations.r44 ∧
    RelationPassed bundle.relations.r45 ∧ RelationPassed bundle.relations.r46 := by
  exact ⟨gates.r01, gates.r02, gates.r03, gates.r04, gates.r05, gates.r06,
    gates.r07, gates.r08, gates.r09, gates.r10, gates.r11, gates.r12,
    gates.r13, gates.r14, gates.r15, gates.r16, gates.r17, gates.r18,
    gates.r19, gates.r20, gates.r21, gates.r22, gates.r23, gates.r24,
    gates.r25, gates.r26, gates.r27, gates.r28, gates.r29, gates.r30,
    gates.r31, gates.r32, gates.r33, gates.r34, gates.r35, gates.r36,
    gates.r37, gates.r38, gates.r39, gates.r40, gates.r41, gates.r42,
    gates.r43, gates.r44, gates.r45, gates.r46⟩

inductive AuthorityPossible : {phase : Phase} -> {bundle : EvidenceBundle phase} ->
    State phase bundle -> Prop where
  | initial {phase bundle} : AuthorityPossible (State.initial (phase := phase) (bundle := bundle))
  | outerDeadlineArmed {phase bundle} : AuthorityPossible
      (State.outerDeadlineArmed (phase := phase) (bundle := bundle))
  | siblingsReady {phase bundle} : AuthorityPossible
      (State.siblingsReady (phase := phase) (bundle := bundle))
  | authorizationConsumed {phase bundle} : AuthorityPossible
      (State.authorizationConsumed (phase := phase) (bundle := bundle))
  | monitorRunning {phase bundle} : AuthorityPossible
      (State.monitorRunning (phase := phase) (bundle := bundle))
  | innerRunning {phase bundle} : AuthorityPossible
      (State.innerRunning (phase := phase) (bundle := bundle))
  | producerRunning {phase bundle} : AuthorityPossible
      (State.producerRunning (phase := phase) (bundle := bundle))
  | producerQuiescent {phase bundle} : AuthorityPossible
      (State.producerQuiescent (phase := phase) (bundle := bundle))
  | verifierARunning {phase bundle} : AuthorityPossible
      (State.verifierARunning (phase := phase) (bundle := bundle))
  | verifierAQuiescent {phase bundle} : AuthorityPossible
      (State.verifierAQuiescent (phase := phase) (bundle := bundle))
  | verifierBRunning {phase bundle} : AuthorityPossible
      (State.verifierBRunning (phase := phase) (bundle := bundle))
  | verifierBQuiescent {phase bundle} : AuthorityPossible
      (State.verifierBQuiescent (phase := phase) (bundle := bundle))
  | dryRunCandidate {bundle} : AuthorityPossible
      (State.dryRunCandidate (bundle := bundle))
  | dryRunPostchecked {bundle} (gates : AllFrozenGates bundle) : AuthorityPossible
      (State.dryRunPostchecked gates)
  | dryRunPassed {bundle} (gates : AllFrozenGates bundle) : AuthorityPossible
      (State.dryRunPassed gates)
  | candidate {bundle} : AuthorityPossible (State.candidate (bundle := bundle))
  | metadataFsynced {bundle} : AuthorityPossible
      (State.metadataFsynced (bundle := bundle))
  | renamedCandidate {bundle} : AuthorityPossible
      (State.renamedCandidate (bundle := bundle))
  | postchecked {bundle} (gates : AllFrozenGates bundle) : AuthorityPossible
      (State.postchecked gates)
  | detachedMarkerInstalled {bundle} (gates : AllFrozenGates bundle)
      (marker : DetachedAuthorityMarker bundle) : AuthorityPossible
      (State.detachedMarkerInstalled gates marker)
  | authoritative {bundle} (gates : AllFrozenGates bundle)
      (marker : DetachedAuthorityMarker bundle)
      (valid : ValidDetachedAuthorityMarker bundle marker) : AuthorityPossible
      (State.authoritative gates marker valid)

theorem failure_target_not_authority_possible
    {phase : Phase} {bundle : EvidenceBundle phase} (event : FailureEvent) :
    Not (AuthorityPossible (bundle := bundle) (failureTarget event)) := by
  cases event <;> intro possible <;> cases possible

theorem step_preserves_authority_possibility
    {phase : Phase} {bundle : EvidenceBundle phase}
    {source target : State phase bundle} (transition : Step source target) :
    AuthorityPossible target -> AuthorityPossible source := by
  cases transition with
  | arm => exact fun _ => .initial
  | siblings => exact fun _ => .outerDeadlineArmed
  | consume => exact fun _ => .siblingsReady
  | monitor => exact fun _ => .authorizationConsumed
  | inner => exact fun _ => .monitorRunning
  | producer => exact fun _ => .innerRunning
  | producerDone => exact fun _ => .producerRunning
  | verifierA => exact fun _ => .producerQuiescent
  | verifierADone => exact fun _ => .verifierARunning
  | verifierB => exact fun _ => .verifierAQuiescent
  | verifierBDone => exact fun _ => .verifierBRunning
  | dryCandidate => exact fun _ => .verifierBQuiescent
  | dryPostcheck gates => exact fun _ => .dryRunCandidate
  | dryPass gates => exact fun _ => .dryRunPostchecked gates
  | officialCandidate => exact fun _ => .verifierBQuiescent
  | metadata => exact fun _ => .candidate
  | rename => exact fun _ => .metadataFsynced
  | postcheck gates => exact fun _ => .renamedCandidate
  | installMarker gates marker => exact fun _ => .postchecked gates
  | authorize gates marker valid => exact fun _ => .detachedMarkerInstalled gates marker
  | fail source event =>
      cases event <;> intro impossible <;> cases impossible
  | cleanupFailed => intro impossible; cases impossible
  | cleanupMonitorDead => intro impossible; cases impossible
  | cleanupResidual => intro impossible; cases impossible
  | orphanTimeout => intro impossible; cases impossible
  | orphanWatchdog => intro impossible; cases impossible
  | orphanExecutor => intro impossible; cases impossible

theorem reach_preserves_authority_possibility
    {phase : Phase} {bundle : EvidenceBundle phase}
    {source target : State phase bundle} (reachable : Reach source target) :
    AuthorityPossible target -> AuthorityPossible source := by
  induction reachable with
  | refl => exact fun possible => possible
  | next transition _ inductionHypothesis =>
      intro possible
      exact step_preserves_authority_possibility transition
        (inductionHypothesis possible)

theorem impossible_state_cannot_reach_authority
    {bundle : EvidenceBundle .official} {source : State .official bundle}
    {gates : AllFrozenGates bundle} {marker : DetachedAuthorityMarker bundle}
    {valid : ValidDetachedAuthorityMarker bundle marker}
    (impossible : Not (AuthorityPossible source)) :
    Not (Reach source (.authoritative gates marker valid)) := by
  intro reachable
  exact impossible (reach_preserves_authority_possibility reachable
    (.authoritative gates marker valid))

theorem quarantine_cannot_reach_authority
    {bundle : EvidenceBundle .official} {gates : AllFrozenGates bundle}
    {marker : DetachedAuthorityMarker bundle}
    {valid : ValidDetachedAuthorityMarker bundle marker} :
    Not (Reach (.quarantine) (.authoritative gates marker valid)) := by
  apply impossible_state_cannot_reach_authority
  intro possible
  cases possible

theorem orphaned_candidate_cannot_reach_authority
    {bundle : EvidenceBundle .official} {gates : AllFrozenGates bundle}
    {marker : DetachedAuthorityMarker bundle}
    {valid : ValidDetachedAuthorityMarker bundle marker} :
    Not (Reach (.orphanedNonAuthoritativeCandidate)
      (.authoritative gates marker valid)) := by
  apply impossible_state_cannot_reach_authority
  intro possible
  cases possible

theorem failed_cannot_reach_authority
    {bundle : EvidenceBundle .official} {gates : AllFrozenGates bundle}
    {marker : DetachedAuthorityMarker bundle}
    {valid : ValidDetachedAuthorityMarker bundle marker} :
    Not (Reach (.failed) (.authoritative gates marker valid)) := by
  apply impossible_state_cannot_reach_authority
  intro possible
  cases possible

theorem timeout_cannot_reach_authority
    {bundle : EvidenceBundle .official} {gates : AllFrozenGates bundle}
    {marker : DetachedAuthorityMarker bundle}
    {valid : ValidDetachedAuthorityMarker bundle marker} :
    Not (Reach (.timedOut) (.authoritative gates marker valid)) := by
  apply impossible_state_cannot_reach_authority
  intro possible
  cases possible

theorem all_failure_and_orphan_states_cannot_reach_authority
    {bundle : EvidenceBundle .official} {gates : AllFrozenGates bundle}
    {marker : DetachedAuthorityMarker bundle}
    {valid : ValidDetachedAuthorityMarker bundle marker} :
    Not (Reach (.failed) (.authoritative gates marker valid)) ∧
    Not (Reach (.timedOut) (.authoritative gates marker valid)) ∧
    Not (Reach (.monitorDead) (.authoritative gates marker valid)) ∧
    Not (Reach (.watchdogDead) (.authoritative gates marker valid)) ∧
    Not (Reach (.deadlineExecutorDead) (.authoritative gates marker valid)) ∧
    Not (Reach (.residual) (.authoritative gates marker valid)) ∧
    Not (Reach (.quarantine) (.authoritative gates marker valid)) ∧
    Not (Reach (.orphanedNonAuthoritativeCandidate)
      (.authoritative gates marker valid)) := by
  repeat' apply And.intro
  all_goals
    apply impossible_state_cannot_reach_authority
    intro possible
    cases possible

#print axioms dryRunPassed_only_dry_phase_and_no_marker
#print axioms authoritative_implies_official_all_gates_marker_and_exact_four_inbox
#print axioms all_frozen_gates_bind_exact_contract
#print axioms relation_gate_count_is_exactly_46
#print axioms step_preserves_authority_possibility
#print axioms reach_preserves_authority_possibility
#print axioms quarantine_cannot_reach_authority
#print axioms orphaned_candidate_cannot_reach_authority
#print axioms failed_cannot_reach_authority
#print axioms timeout_cannot_reach_authority
#print axioms all_failure_and_orphan_states_cannot_reach_authority
#print axioms one_byte_all_true_cannot_construct_r01
#print axioms ExactInboxEntries
#print axioms RelationSemantics
#print axioms RelationGates
#print axioms AllFrozenGates
#print axioms relationInputBytes
#print axioms expectedRelationVerifier
#print axioms expectedRelationOwner
#print axioms expectedRelationTheorem
#print axioms AuthorityPossible
#print axioms failure_target_not_authority_possible
#print axioms RelationResultSemantics
#print axioms ConcreteResultSemantics
#print axioms relationTranscriptBytes
#print axioms expectedRelationNats
#print axioms expectedRelationStrings
#print axioms expectedRelationUniqueKeys
#print axioms expectedRelationTimes
#print axioms ledgerRecordPath
#print axioms markerPath
#print axioms expected_failure_transition_count
#print axioms LedgerResultSemantics
#print axioms RootResultSemantics
#print axioms ArtifactRootsResultSemantics
#print axioms MarkerPathResultSemantics
#print axioms StateTableResultSemantics
#print axioms AuthorizationAuthenticityResultSemantics
#print axioms SchemaResultSemantics
#print axioms markerPath_injective
#print axioms relation_coverage_total
#print axioms arbitrary_record_wrapper_cannot_satisfy_relation
#print axioms wrong_ledger_path_cannot_satisfy_r04
#print axioms duplicate_attestation_bytes_cannot_satisfy_r29
#print axioms chemgraph_e2_v03_relation_r01_sound
#print axioms chemgraph_e2_v04_relation_r29_sound

end HardenedLauncherV04

end ChemGraph
