from ego_mol_llm.validate import parse_model_output, expand_smiles_field

raw = '{"smiles": "[' + "'CCO','c1ccccc1O'" + ']", "adduct": "[M+H]+", "confidence": 0.5}'
# cleaner:
raw = """{"smiles": ["CCO", "c1ccccc1O"], "adduct": "[M+H]+", "confidence": 0.5}"""
p = parse_model_output(raw, precursor_mz=95.0491, mass_tol_da=0.05)
print("list_json", p.smiles, p.smiles_valid, p.mass_ok, p.matched_adduct)

raw2 = """{"smiles": "['CCO', 'c1ccccc1O']", "adduct": "[M+H]+"}"""
p2 = parse_model_output(raw2, precursor_mz=95.0491, mass_tol_da=0.05)
print("list_str", p2.smiles, p2.smiles_valid, p2.mass_ok, p2.matched_adduct)

raw3 = '{"smiles": "c1ccccc1O", "adduct": "[M+H]+"}'
p3 = parse_model_output(raw3, precursor_mz=95.0491, mass_tol_da=0.05)
print("phenol", p3.smiles, p3.mass_ok, p3.matched_adduct)

print("expand", expand_smiles_field("['CC','CCO','c1ccccc1']"))
print("ok")
