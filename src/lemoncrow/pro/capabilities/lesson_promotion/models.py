"""Re-export of typed adaptive lesson models.

Defined open in ``lesson_promotion_contract`` (data contract, not IP; pydantic
cannot be mypyc-compiled). Kept importable at the original path for the compiled
pro logic.
"""

from __future__ import annotations

from lemoncrow.core.capabilities.lesson_promotion_contract import (
    CostCapBreachMode as CostCapBreachMode,
)
from lemoncrow.core.capabilities.lesson_promotion_contract import (
    LessonScope as LessonScope,
)
from lemoncrow.core.capabilities.lesson_promotion_contract import (
    TypedLesson as TypedLesson,
)
from lemoncrow.core.capabilities.lesson_promotion_contract import (
    TypedLessonKind as TypedLessonKind,
)

__all__ = ["CostCapBreachMode", "LessonScope", "TypedLesson", "TypedLessonKind"]
