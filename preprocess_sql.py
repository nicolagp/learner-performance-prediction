import pandas as pd
import psycopg2
import argparse

def get_students(n, conn):
    q = f"""
    SELECT id
    FROM enem_student
    ORDER BY RANDOM()
    LIMIT {n};
    """
    return list(pd.read_sql_query(q, conn)["id"].values)


def get_questions(student_id, conn):
    """
    Returns a dataframe with questions answered by a student
    Needs a student id and a connection to the db
    """
    q = f"""
    SELECT student_id, question_id, letter
    FROM enem_questionstudent
    WHERE student_id = {student_id}
    ORDER BY RANDOM();
    """
    return pd.read_sql_query(q, conn)

def get_answers(conn):
    q = """
    SELECT id, correct, skill_id
    FROM enem_question
    """
    gabarito = pd.read_sql_query(q, conn)
    gabarito = gabarito.set_index("id")
    return gabarito

def correct(x, gabarito):
    """
    given that a 'questions' answer key is defined, it checks wether a student's answer is correct
    """
    return 1 if x[2] == gabarito.loc[x[1], "correct"] else 0

def pipeline(students, conn):
    """
    Takes in a list of student ids and outputs a dataframe in the correct format
    """
    gabarito = get_answers(conn)
    
    df = pd.DataFrame(columns = ["student_id", "question_id", "letter", "correct", "skill_id"])

    for student in students:
        # get student answers
        stu = get_questions(student, conn)
        
        # handle missing data
        if stu.empty:
            continue
            
        # correct col
        stu["correct"] = stu.apply(correct, axis=1, args=(gabarito,))

        # skill_id col
        stu["skill_id"] = gabarito.loc[stu["question_id"], "skill_id"].values

        # concat df
        df = pd.concat((df, stu), axis=0)
    
    # rename cols
    df = df.rename({"student_id": "user_id", "question_id": "item_id"}, axis=1)
    
    # add timestamp
    df["timestamp"] = 0
    
    # return columns in correct order
    cols = ["user_id", "item_id", "timestamp", "correct", "skill_id"]
    return df.loc[:, cols]


def main(args):
    # create connection
    conn = psycopg2.connect(host="localhost",
                            port = 5432,
                            database="aio_enem",
                            user="nicolagp",
                            password="123")

    # get student list
    n = args.n
    students = get_students(n, conn)

    # get data from pipeline
    df = pipeline(students, conn)

    # save data to csv
    train_size = args.train_size
    df_size = df.shape[0]
    fn = "./data/2015_CN_AZUL/preprocessed_data.csv"
    fn_train = "./data/2015_CN_AZUL/preprocessed_data_train.csv"
    fn_test = "./data/2015_CN_AZUL/preprocessed_data_test.csv"

    df.to_csv(fn, sep="\t", index=False)
    df.iloc[:int(train_size*df_size)].to_csv(fn_train, sep="\t", index=False)
    df.iloc[int(train_size*df_size):].to_csv(fn_test, sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SQL Preprocessing')
    parser.add_argument("-n", type=int, help="Number of students to use")
    parser.add_argument("--train_size", type=float, help="Fraction of data\
to be used for the train set: [0,1]")
    args = parser.parse_args()

    main(args)


