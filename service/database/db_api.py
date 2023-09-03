import sqlite3 as db

class DBAPI:
    """
    Класс обращение к БД на базе sqlite
    """
    def __init__(self):
        self.con = db.connect("service/database/datamart.db")
        self.create_table()

    def __del__(self):
        self.con.close()

    def create_table(self):
        sql = """CREATE TABLE IF NOT EXISTS ..."""
        cur = self.con.cursor()
        cur.execute(sql)

