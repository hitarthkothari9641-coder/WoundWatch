"""Local administrator commands. Passwords are prompted, never command arguments."""
import argparse
import getpass
import json
from . import store
from .accounts import create_user,password_hash,audit

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    create=sub.add_parser('create-user');create.add_argument('--email',required=True);create.add_argument('--name',required=True)
    create.add_argument('--role',choices=['patient','clinician'],required=True);create.add_argument('--patient-id')
    assign=sub.add_parser('assign');assign.add_argument('--email',required=True);assign.add_argument('--patient-id',required=True)
    revoke=sub.add_parser('unassign');revoke.add_argument('--email',required=True);revoke.add_argument('--patient-id',required=True)
    reset=sub.add_parser('reset-password');reset.add_argument('--email',required=True)
    sub.add_parser('list-records')
    args=ap.parse_args()
    if args.command=='list-records':
        for p in store.records('patients'):print(p['id'],p['name'])
        return
    if args.command in ('create-user','reset-password'):
        password=getpass.getpass('New password (12–128 characters): ')
        if password!=getpass.getpass('Confirm password: '):ap.error('Passwords do not match')
        if not 12<=len(password)<=128:ap.error('Password must be 12–128 characters')
        if args.command=='create-user':
            user=create_user(args.email,args.name,password,args.role,args.patient_id);print('Created',user['role'],'account',user['id']);return
    with store.connection() as db:
        user=db.execute('SELECT * FROM users WHERE email=?',(args.email.lower().strip(),)).fetchone()
        if not user:ap.error('Account not found')
        if args.command=='reset-password':
            db.execute('UPDATE users SET password_hash=? WHERE id=?',(password_hash(password),user['id']))
            db.execute('DELETE FROM sessions WHERE user_id=?',(user['id'],));audit(db,'admin_password_reset',user['id'])
        else:
            if user['role']!='clinician':ap.error('Assignments require a clinician account')
            if not db.execute('SELECT id FROM patients WHERE id=?',(args.patient_id,)).fetchone():ap.error('Patient record not found')
            if args.command=='assign':db.execute('INSERT OR IGNORE INTO assignments VALUES(?,?)',(user['id'],args.patient_id))
            else:db.execute('DELETE FROM assignments WHERE user_id=? AND patient_id=?',(user['id'],args.patient_id))
            audit(db,args.command,args.patient_id)
    print('Completed',args.command)

if __name__=='__main__':main()
