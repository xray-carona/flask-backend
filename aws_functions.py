import boto3
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET,AWS_CT_FOLDER

session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
s3_resource = session.resource('s3', region_name=AWS_REGION)
s3_bucket = s3_resource.Bucket(S3_BUCKET)


def get_xray_image(url):
    urls = url.split('/')
    folder, file = urls[3]+'/'+urls[-2], urls[-1]
    xray = s3_bucket.Object(folder + "/" + file)
    response = xray.get()
    file_stream = response['Body']
    return file_stream


def return_s3_url_image(filename,folder):
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{folder}{filename}.jpg"


def upload_to_s3(image, file_name):
    upload_image = s3_bucket.Object(key=f"{AWS_CT_FOLDER}{file_name}.jpg")
    upload_image.put(Body=image,ACL='public-read')
    return return_s3_url_image(file_name,AWS_CT_FOLDER)


if __name__=="__main__":
    pass
