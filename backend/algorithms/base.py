class BaseDispatcher:
    def choose_action(self, building) -> int:
        raise NotImplementedError