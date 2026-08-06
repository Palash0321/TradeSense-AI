from app.auth.jwt_handler import (
    create_access_token,
    verify_access_token,
)

token = create_access_token(
    {
        "sub": "palash@example.com"
    }
)

print()

print("TOKEN")

print(token)

print()

print("VERIFY")

print(verify_access_token(token))