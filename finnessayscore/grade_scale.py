def mk_grade(name):
    if name == "fivehigh":
        return {"lab_grade": [str(grade) for grade in range(1, 6)]}
    elif name == "outof20":
        return {"lab_grade": [str(grade) for grade in range(21)]}


DEFAULT_GRADE_SCALE = mk_grade("fivehigh")
