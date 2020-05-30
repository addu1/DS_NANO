from sqlalchemy import create_engine
engine = create_engine('sqlite:///DisasterResponse.db')
print (engine.table_names()[0].split('.')[0])