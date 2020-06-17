import pandas as pd
import psycopg2
import argparse
import DB_CONFIG

def get_students(n, conn):
    q = f"""
    SELECT id
    FROM enem_student
    ORDER BY RANDOM()
    LIMIT {n};
    """
    return list(pd.read_sql_query(q, conn)["id"].values)

def get_question_student(students, inep_questions, conn):
    """
    Returns a dataframe with questions answered by a student
    Needs a list of students, a list of questions that are in the desired area
    and a connection to the db.
    """
    q = f"""
    SELECT student_id, letter, inep_code, skill_id
    FROM enem_questionstudent
        JOIN enem_question
        ON enem_question.id = enem_questionstudent.question_id
    WHERE student_id IN {tuple(students)} AND inep_code IN {tuple(inep_questions)}
    ORDER BY student_id ASC, RANDOM();
    """
    return pd.read_sql_query(q, conn)

def get_answers(inep_questions, conn):
    q = f"""
    SELECT inep_code, correct, skill_id
    FROM enem_question
    WHERE inep_code IN {tuple(inep_questions)}
    """
    gabarito = pd.read_sql_query(q, conn)
    gabarito = gabarito.set_index("inep_code")
    return gabarito

def get_question_ids(area, conn):
    # get sub exams that are on a given area
    q = """
    SELECT id
    FROM enem_subexam
    WHERE area = 'CN'
    """
    se_ids = tuple(pd.read_sql_query(q, conn).values.flatten())

    # get question ids from area
    q = f"""
        SELECT DISTINCT(inep_code)
        FROM enem_subexamquestion
            JOIN enem_question
            ON enem_subexamquestion.question_id = enem_question.id
        WHERE sub_exam_id IN {se_ids}
        """
    return pd.read_sql_query(q, conn).values.flatten()

def correct(x, answers):
    """
    given that a 'questions' answer key is defined, it checks wether a student's answer is correct
    """
    return 1 if x[1] == answers.loc[x[2], "correct"] else 0

def pipeline(args, conn):
    """
    Takes in arguments and outputs a dataframe in the correct format
    """
    # get students to parse
    students = get_students(args.n, conn)

    # get id of questions that match selected area
    inep_questions = get_question_ids(args.area, conn)

    # get answers for questions
    answers = get_answers(inep_questions, conn)

    # get df of students with selected questions and answers
    df = get_question_student(students, inep_questions, conn)

    # calculate correctness
    df["correct"] = df.apply(correct, axis=1, args=(answers,))

    # select and rename columns
    df["timestamp"] = 0
    df = df.rename({"student_id": "user_id", "inep_code": "item_id"}, axis=1)
    cols = ["user_id", "item_id", "timestamp", "correct", "skill_id"]
    
    return df.loc[:, cols]



def main(args):
    # create connection
    # conn = psycopg2.connect(host="localhost",
    #                         port = 5432,
    #                         database="aio_enem",
    #                         user="nicolagp",
    #                         password="123")

    conn = psycopg2.connect(host=DB_CONFIG.host,
                            port = DB_CONFIG.port,
                            database=DB_CONFIG.database,
                            user=DB_CONFIG.user,
                            password=DB_CONFIG.password)

    # get data from pipeline
    df = pipeline(args, conn)

    # save data to csv
    train_size = args.train_size
    df_size = df.shape[0]
    if args.test:
        fn = "./data/test/preprocessed_data.csv"
        fn_train = "./data/test/preprocessed_data_train.csv"
        fn_test = "./data/test/preprocessed_data_test.csv"
    else:
        fn = "./data/2015_CN_AZUL/preprocessed_data.csv"
        fn_train = "./data/2015_CN_AZUL/preprocessed_data_train.csv"
        fn_test = "./data/2015_CN_AZUL/preprocessed_data_test.csv"

    df.to_csv(fn, sep="\t", index=False)
    df.iloc[:int(train_size*df_size)].to_csv(fn_train, sep="\t", index=False)
    df.iloc[int(train_size*df_size):].to_csv(fn_test, sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SQL Preprocessing')
    parser.add_argument("-n", type=int, help="Number of students to use")
    parser.add_argument("--area", type=str, help="Area of knowledge [CN, CH, LC, MT]")
    parser.add_argument("--test", type=bool, help="Wether to save the data to a test directory")
    parser.add_argument("--train_size", type=float, help="Fraction of data\
to be used for the train set: [0,1]")
    args = parser.parse_args()

    # validate arguments
    if args.area not in ["CN", "CH", "LC", "MT"]:
        raise "Invalid Area, expecting one of [CN, CH, LC, MT]"
    if args.train_size < 0 or args.train_size > 1:
        raise "Invalid train_size, should be in the interval [0,1]"

    main(args)


