def engagement_score(face, posture, gestures):

    score = 0

    if face:
        score += 35

    if posture == "Open":
        score += 30

    if gestures > 20:
        score += 20

    return min(score,100)