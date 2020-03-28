import os
import psycopg2
# cnx_string=os.getenv('postgre')
conn = psycopg2.connect(host="localhost",database="suppliers", user="postgres", password="postgres")

def write_output_to_db():
    pass