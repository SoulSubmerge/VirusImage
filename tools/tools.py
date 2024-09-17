def inferType(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    
    try:
        return float(value)
    except ValueError:
        pass
    
    return value



def evaluateRule(rule, variables)->bool:
    return eval(rule,{}, variables)