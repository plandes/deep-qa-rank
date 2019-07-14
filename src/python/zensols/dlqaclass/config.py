from zensols.actioncli import ExtendedInterpolationConfig


class AppConfig(ExtendedInterpolationConfig):
    LAST_INST = None

    def __init__(self, *args, **kwargs):
        super(AppConfig, self).__init__(*args, **kwargs, default_expect=True)
        self.__class__.LAST_INST = self

    @classmethod
    def instance(cls):
        if cls.LAST_INST is None:
            raise ValueError('configuration has never been created')
        return cls.LAST_INST
