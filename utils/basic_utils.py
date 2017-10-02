import pickle
import logging
import smtplib
import traceback
from email.mime.text import MIMEText

from .config import MailSettings as MS


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def send_mail():
    me = "mengnan" + "<" + MS.MAIL_USER + ">"
    content = 'Hurry up! Click here to SHUTDOWN the instance: https://console.aws.amazon.com/ec2/v2/home?region=us-east-1'
    msg = MIMEText(content, _subtype='plain', _charset='utf-8')
    msg['Subject'] = "The model training is finished"
    msg['From'] = me
    msg['To'] = ";".join(MS.MAILTO_LIST)
    try:
        s = smtplib.SMTP(MS.MAIL_HOST)
        s.ehlo()
        s.starttls()
        s.login(MS.MAIL_USER, MS.MAIL_PASS)
        s.sendmail(me, MS.MAILTO_LIST, msg.as_string())
        s.close()
        print('Then notification email has been sent.')
    except Exception as e:
        print(traceback.format_exc())


if __name__ == '__main__':
    send_mail()
