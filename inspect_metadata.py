
from apps.orchestrator.models import Base
print("Tables in Base.metadata:")
for table_name in Base.metadata.tables:
    print(f"Table: {table_name}")
    table = Base.metadata.tables[table_name]
    print(f"  Columns: {[c.name for c in table.columns]}")

if 'workflow_runs' in Base.metadata.tables:
    columns = [c.name for c in Base.metadata.tables['workflow_runs'].columns]
    if 'completed_applets' in columns:
        print("SUCCESS: completed_applets is in Base.metadata")
    else:
        print("FAILURE: completed_applets is NOT in Base.metadata")
else:
    print("FAILURE: workflow_runs table NOT found in Base.metadata")
