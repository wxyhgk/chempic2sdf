"""No artifact/checkpoint reads: synthetic compiler attribution contracts."""
import torch

from tools.audit_r17_selector_graph_replay import compile_prediction, strict_identity


class FixedHead(torch.nn.Module):
    def __init__(self, classes, selected=0):
        super().__init__()
        self.classes, self.selected = classes, selected

    def forward(self, x):
        result = x.new_zeros((len(x), self.classes))
        result[:, self.selected] = 10
        return result


class AromaticHead(torch.nn.Module):
    def forward(self, x):
        assert x.shape[-1] == 526
        return x.new_full((len(x),), 10)


def fixture(slots):
    n = max(slots) + 1
    raw = {'atom_geometry':torch.zeros(1,n,2),
           'atom_features':torch.zeros(1,n,128),'atom_scores':torch.ones(1,n)}
    mask = torch.zeros(n,dtype=torch.bool); mask[slots] = True
    p = len(slots)*(len(slots)-1)//2
    direct = {'hidden':torch.zeros(p,256),'presence_logits':torch.full((p,),10.),
              'chemical_logits':torch.zeros(p,4)}
    return raw,mask,direct


def test_sparse_slots_do_not_silently_drop_compiler_edges():
    raw,mask,direct = fixture([4,7])
    ok,error,pred = compile_prediction(raw,direct,mask,FixedHead(4,2),FixedHead(11,1),
                                      AromaticHead(),.5,[4,7])
    # O triple O exceeds the neutral-O valence. Ignoring raw slots would
    # incorrectly compile two isolated oxygens successfully.
    assert not ok and error == 'valence'
    assert pred['edges'] == [[4,7,3]]


def test_aromatic_cycle_remains_aromatic_at_compiler_boundary():
    raw,mask,direct = fixture([2,4,7])
    _,_,pred = compile_prediction(raw,direct,mask,FixedHead(4),FixedHead(11),
                                  AromaticHead(),.5,[2,4,7])
    assert all(edge[2] == 4 for edge in pred['edges'])


def test_unmapped_edge_counts_as_false_positive_and_type_is_scored():
    raw,_,_ = fixture([1,3,5])
    raw['atom_geometry'][0,1] = torch.tensor([.1,.1])
    raw['atom_geometry'][0,3] = torch.tensor([.2,.1])
    raw['atom_geometry'][0,5] = torch.tensor([.8,.8])
    centers = torch.tensor([[.1,.1],[.2,.1]])
    pred = {'slots':[1,3,5],'elements':['C','O','C'],'edges':[[1,3,1],[3,5,1]]}
    result = strict_identity(pred,raw,centers,[{'element':'C'},{'element':'O'}],{(0,1):'DOUBLE'})
    assert result['predicted_edges'] == 2 and result['unmapped_edges'] == 1
    assert result['true_edges'] == 1 and result['typed_true_edges'] == 0
    assert not result['exact_elements_and_typed_topology']
