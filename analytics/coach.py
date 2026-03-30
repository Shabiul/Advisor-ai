def generate_coaching(insights):

    advice = []

    for insight in insights:

        if "eye contact" in insight:
            advice.append("Try asking a direct question to regain attention")

        if "Closed posture" in insight:
            advice.append("Build rapport before pushing further")

        if "Low hand activity" in insight:
            advice.append("Encourage interaction with open-ended questions")

        if "low" in insight:
            advice.append("Change tone or topic to re-engage client")

    if not advice:
        advice.append("Maintain current approach")

    return advice