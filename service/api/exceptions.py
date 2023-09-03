import typing as tp
from http import HTTPStatus


class AppException(Exception):
    def __init__(
            self,
            status_code: int,
            error_key: str,
            error_message: str = "",
            error_loc: tp.Optional[tp.Sequence[str]] = None,
    ) -> None:
        self.error_key = error_key
        self.error_message = error_message
        self.error_loc = error_loc
        self.status_code = status_code
        super().__init__()


class UserNotFoundError(AppException):
    def __init__(
            self,
            status_code: int = HTTPStatus.NOT_FOUND,
            error_key: str = "user_not_found",
            error_message: str = "User is unknown",
            error_loc: tp.Optional[tp.Sequence[str]] = None,
    ):
        super().__init__(status_code, error_key, error_message, error_loc)


class FileNotFoundError(AppException):
    """
    Исключение при обращении не к той дкларации
    """

    def __init__(
            self,
            status_code: int = HTTPStatus.NOT_FOUND,
            error_key: str = "file_not_found",
            error_message: str = "File not found",
            error_loc: tp.Optional[tp.Sequence[str]] = None,
    ):
        super().__init__(status_code, error_key, error_message, error_loc)


class NotAuthorizedError(AppException):
    """
    Исключение при обращении без токена
    """

    def __init__(
            self,
            status_code: int = HTTPStatus.UNAUTHORIZED,
            error_key: str = "authorisation_failed",
            error_message: str = "Authorization failed! API token is not properly set",
            error_loc: tp.Optional[tp.Sequence[str]] = None,
    ):
        super().__init__(status_code, error_key, error_message, error_loc)
