path = r'C:\Users\alisa\DS-ML-DL\Fabric-Analytics-Projects\ws_Finance_Analysis\sm_Finance.SemanticModel\definition\expressions.tmdl'

lines = [
    "expression 'DirectLake - lh_Finance_Silver' =",
    "\t\tlet",
    "\t\t    Source = AzureStorage.DataLake(\"https://onelake.dfs.fabric.microsoft.com/61549e76-c4d4-4b27-9018-ab9b04eab5dc/d726b863-4ac2-4206-85ac-409c87da9f22\", [HierarchicalNavigation=true])",
    "\t\tin",
    "\t\t    Source",
    "\tlineageTag: b1c2d3e4-f5a6-7890-bcde-fa1234567801",
    "",
    "\tannotation PBI_IncludeFutureArtifacts = False",
    ""
]

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(lines))

with open(path, 'r', encoding='utf-8') as f:
    print(f.read())
print('Done')
