import os
import psycopg2

postgres_host = os.getenv('POSTGRES_HOST')
postgres_db = os.getenv('POSTGRES_DB')
postgres_user = os.getenv('POSTGRES_USER')
postgres_password = os.getenv('POSTGRES_PASSWORD')
conn = psycopg2.connect(host=postgres_host, database=postgres_db, user=postgres_user, password=postgres_password)


def write_output_to_db(params_dict):
    cursor = conn.cursor()
    query = "INSERT INTO xray_results(img_url,model_version,model_output) VALUES (%(img_url)s,%(model_version)s,%(model_output)s)"
    cursor.execute(query, params_dict)
    conn.commit()
    cursor.close()


def setup_db():
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS xray_results (
    id serial primary key ,
    img_url text not null ,
    model_version varchar (50) not null ,
    model_output varchar(100) not null,
    gender varchar(10),
    age int,
    created_at timestamp default CURRENT_TIMESTAMP not null,
    user_id int
);
    """)
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    setup_db()
