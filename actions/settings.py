from sklearn.tree import DecisionTreeClassifier

class Settings:
    feature_names = None
    test_size = None
    model = None
    parameters = None
    cv = None

    def __init__(self):
        # 디폴트 초기설정들....DT로 정하자...
        self.feature_names = ["Pclass", "Sex", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"]
        self.model = DecisionTreeClassifier(max_depth=5, random_state=37)
        self.parameters = {"max_depth":5, "random_state":37}
        self.cv = 10
        self.test_size = 0.3

    def set_model(self, model, parameters = None, cv=None):
        self.model = model
        self.parameters = parameters
        if cv != None:
            self.cv = cv

