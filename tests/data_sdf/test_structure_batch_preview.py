from PIL import Image

from tools.preview_structure_batch import choose_rows, write_pages


def test_selection_uses_only_retained_and_never_substitutes_missing_identity():
    selection=[dict(molecule_id='a'),dict(molecule_id='b')]
    rows=[dict(molecule_id='a',sample_key='a_00',profile_id='oa00'),
          dict(molecule_id='a',sample_key='a_22',profile_id='oa22'),
          dict(molecule_id='b',sample_key='b_22',profile_id='oa22')]
    assert choose_rows(selection,rows,{'a_00'})==[(selection[0],rows[0]),(selection[1],None)]
    assert choose_rows(selection,rows,{'a_00','a_22'})[0][1]==rows[1]


def test_contact_sheet_is_separate_and_does_not_change_original(tmp_path):
    root=tmp_path/'data'; (root/'samples').mkdir(parents=True)
    original=root/'samples'/'a.png'; Image.new('RGB',(768,768),'white').save(original)
    before=original.read_bytes(); output=tmp_path/'review'; output.mkdir()
    selection=[(dict(molecule_id='a',concepts=['test'],canonical_smiles='CC'),dict(sample_key='a',profile_id='oa22'))]
    pages=write_pages(root,output,selection)
    assert pages==['structures-01.png'] and original.read_bytes()==before
    with Image.open(output/pages[0]) as image: assert image.size==(1200,1090)
