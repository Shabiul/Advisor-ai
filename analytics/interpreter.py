def interpret_behavior(data):

    insights = []

    if not data["eye_contact"]:
        insights.append("Client is not maintaining eye contact → possible disengagement")

    if data["posture"] == "Closed":
        insights.append("Closed posture detected → resistance or discomfort")

    if data["gestures"] < 10:
        insights.append("Low hand activity → low engagement or hesitation")

    if data["engagement"] < 50:
        insights.append("Overall engagement is low")

    if not insights:
        insights.append("Client is engaged and responsive")

    return insights