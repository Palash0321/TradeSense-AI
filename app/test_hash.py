from app.auth.hashing import (
    hash_password,
    verify_password
)

password = "TradeSense123"

hashed = hash_password(password)

print("Hashed Password:")
print(hashed)

print()

print(
    "Password Match:",
    verify_password(password, hashed)
)