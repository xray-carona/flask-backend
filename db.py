from config import POSTGRES_HOST, POSTGRES_DB, POSTGRES_PASSWORD, POSTGRES_USER
import psycopg2


def connect_db():
    conn = psycopg2.connect(host=POSTGRES_HOST, database=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD)
    return conn


def write_output_to_db(params_dict):
    conn = connect_db()
    cursor = conn.cursor()
    query = "INSERT INTO xray_results(img_url,model_version,model_output) VALUES (%(img_url)s,%(model_version)s,%(model_output)s)"
    cursor.execute(query, params_dict)
    conn.commit()
    cursor.close()
    conn.close()


def setup_db():
    conn = connect_db()
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
