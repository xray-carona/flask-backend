from config import POSTGRES_HOST, POSTGRES_DB, POSTGRES_PASSWORD, POSTGRES_USER
import psycopg2


def connect_db():
    conn = psycopg2.connect(host=POSTGRES_HOST, database=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD)
    return conn


def write_output_to_db(params_dict):
    conn = connect_db()
    cursor = conn.cursor()
    query = """INSERT INTO xray_results(user_id,img_url,model_version,model_output,patient_info,input_image_hash)
     VALUES (%(user_id)s,%(img_url)s,%(model_version)s,%(model_output)s,%(patient_info)s,%(input_image_hash)s)"""
    cursor.execute(query, params_dict)
    conn.commit()
    cursor.close()
    conn.close()


def get_model_output(params_dict):
    conn = connect_db()
    cursor = conn.cursor()
    args = (str(params_dict),)
    query = "SELECT model_output FROM xray_results WHERE input_image_hash=%(image_hash)s and model_version=%(model_version)s;"
    #        #print(query % tuple(args))
    cursor.execute(query, params_dict)
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result


def setup_db():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS xray_results (
    id serial primary key ,
    img_url text not null ,
    model_version varchar (50) not null ,
    model_output jsonb not null,
    gender varchar(10),
    age int,
    created_at timestamp default CURRENT_TIMESTAMP not null,
    user_id int,
    patient_id int
);
    """)
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    setup_db()
