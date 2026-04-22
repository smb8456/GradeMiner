# -----------------------------------------------------------------------------------------------------------------------------
# GradeMiner
# CMPSC 446
# student.py | Represents a single student record from the UCI Student Performance dataset.
# -----------------------------------------------------------------------------------------------------------------------------


from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Student:
    """
    Stores demographic, social, and academic attributes for one student.
    Matches the UCI Student Performance dataset columns.
    """
    # School & demographics
    school: str = ""
    sex: str = ""
    age: int = 0
    address: str = ""
    famsize: str = ""
    Pstatus: str = ""

    # Parental education & jobs
    Medu: int = 0
    Fedu: int = 0
    Mjob: str = ""
    Fjob: str = ""

    # School choice & support
    reason: str = ""
    guardian: str = ""
    traveltime: int = 0
    studytime: int = 0
    failures: int = 0
    schoolsup: str = ""
    famsup: str = ""
    paid: str = ""
    activities: str = ""
    nursery: str = ""
    higher: str = ""
    internet: str = ""
    romantic: str = ""

    # Social & lifestyle
    famrel: int = 0
    freetime: int = 0
    goout: int = 0
    Dalc: int = 0
    Walc: int = 0
    health: int = 0
    absences: int = 0

    # Grades
    G1: int = 0
    G2: int = 0
    G3: int = 0  # Final grade — prediction target

    # Derived label (set after loading)
    pass_fail: Optional[str] = field(default=None, repr=False)

    def set_pass_fail(self, threshold: int = 10) -> None:
        """Label this student as Pass or Fail based on G3."""
        self.pass_fail = "Pass" if self.G3 >= threshold else "Fail"

    def __str__(self) -> str:
        return (
            f"Student | Age: {self.age}, Sex: {self.sex}, "
            f"G1: {self.G1}, G2: {self.G2}, G3: {self.G3}, "
            f"Label: {self.pass_fail}"
        )
