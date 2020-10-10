# -*- coding: utf-8 -*-
"""
@author: hugonnet
Python routines to extract aster l1a LPDAAC pulldirs from emails (mbox file) into a list for wget

2 choices:
- export to a .mbox manually and read
- read everything through GMAIL directly (forwarded mail might need some adapting of the "str.find")

to export into a .mbox:
GMAIL : put in specific folder or under specific label and use https://takeout.google.com/settings/takeout
THUNDERBIRD : use ImportExport AddOn: https://addons.mozilla.org/en-US/thunderbird/addon/importexporttools/

for gmail:
need to put everything under a label and authorize access to "less secure apps" in security
"""

# TODO: script MBOX only works in Python 2.7 apparently
from __future__ import print_function
import mailbox
import imaplib
import email
import csv


def getbody(message):  # getting plain text 'email body'
    body = None
    if message.is_multipart():
        for part in message.walk():
            if part.is_multipart():
                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        body = subpart.get_payload(decode=True)
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
    elif message.get_content_type() == 'text/plain':
        body = message.get_payload(decode=True)
    return body


def read_email_from_mbox(in_mbox):
    print('Reading MBOX file...')
    mbox = mailbox.mbox(in_mbox)
    listPullDir = list()

    print('IDing PullDirs links...')
    for message in mbox:  # looping through stacked messages
        tmp = getbody(message)
        pos1 = str.find(tmp, 'https://e4ftl01.cr.usgs.gov/PullDir/')  # search root link in email
        pos2 = str.find(tmp[pos1:len(tmp)],
                        '\r\nDownload ZIP file of packaged order:')  # search first end in remaining message

        PullDir = tmp[pos1:pos1 + pos2]
        listPullDir.append(PullDir)

    return listPullDir


def read_email_from_gmail(label):
    ORG_EMAIL = "@gmail.com"
    FROM_EMAIL = "username" + ORG_EMAIL #input username
    FROM_PWD = "password" #input password
    SMTP_SERVER = "imap.gmail.com"
    SMTP_PORT = 993
    try:
        mail = imaplib.IMAP4_SSL(SMTP_SERVER)
        mail.login(FROM_EMAIL, FROM_PWD)
        mail.select(label)

        type, data = mail.search(None, 'ALL')
        mail_ids = data[0]

        id_list = mail_ids.split()
        first_email_id = int(id_list[0])
        latest_email_id = int(id_list[-1])

        listPullDir = []

        print('Number of emails: ' + str(latest_email_id))
        for i in range(latest_email_id, first_email_id - 1, -1):
            print('Email number ', i, 'out of', latest_email_id)
            typ, data = mail.fetch(i, '(RFC822)')

            for response_part in data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_string(response_part[1])
                    tmp = msg.get_payload()
                    # !! search criteria might change with time depending on LPDAAC email naming convention!!

                    pos1 = str.find(tmp, 'https://e4ftl01.cr.usgs.gov/PullDir/')  # search root link in email

                    pos2 = str.find(tmp[pos1:len(tmp)],
                                    '\r\nDownload ZIP file of packaged order:')  # search first end in remaining message

                    # added this for change in mail format on 09/2019...
                    if pos2 == -1:
                        pos2 = str.find(tmp[pos1:len(tmp)], '\r\n\r\nExpiration:')

                        PullDir = tmp[pos1:pos1 + pos2]
                    else:
                        PullDir = tmp[pos1+12:pos1 + pos2]

                    listPullDir.append(PullDir)

                    PullDir = tmp[pos1:pos1 + pos2]
                    listPullDir.append(PullDir)

    except Exception:
        print(str(Exception))

    return listPullDir


def writelistPullDir(out_list, listPullDir):
    print('Writing links to csv list ' + out_list + '...')
    with open(out_list, 'wd') as file:
        writer = csv.writer(file, delimiter=',')
        for dir in listPullDir:
            writer.writerow([dir])


if __name__ == '__main__':
    # mbox solution
    in_mbox = '/home/atom/proj/ww_tvol_study/worldwide/global/L1A_retrieval/LPDAAC_ww2_2.mbox'
    out_csv = '/home/atom/proj/ww_tvol_study/worldwide/global/L1A_retrieval/list_PullDirs_ww2_2.csv'
    listdir = read_email_from_mbox(in_mbox)
    writelistPullDir(out_csv, listdir)

    # gmail solution
    label_name = 'LPDAAC_mails'
    listdir = read_email_from_gmail(label_name)
    writelistPullDir(out_csv, listdir)
