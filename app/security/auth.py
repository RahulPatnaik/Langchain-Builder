from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from loguru import logger

from app.config import settings

# --- API Key Authentication ---

API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """
    Dependency to validate the API key provided in the X-API-KEY header.
    """
    if not api_key:
        logger.warning("API Key missing from request.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing",
            headers={"WWW-Authenticate": "API Key"},
        )
    if api_key != settings.API_KEY:
        logger.warning("Invalid API Key received.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "API Key"},
        )
    # Optional: Could log successful validation, but might be noisy
    # logger.debug("API Key validated successfully.")
    return api_key # Return the key itself if needed, or just return True


# --- OAuth2 Password Bearer (Placeholder/Example) ---
# This requires a /token endpoint and JWT implementation

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token") # Adjust tokenUrl as needed

# This is a *placeholder* dependency. Replace with actual JWT verification.
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Placeholder dependency for verifying OAuth2 Bearer token (JWT).
    Replace with actual JWT decoding and user lookup.
    """
    logger.debug(f"Received token for verification (placeholder): {token[:10]}...")
    # --- Replace this with your actual JWT verification logic ---
    # Example using python-jose (install required):
    # from jose import jwt, JWTError
    # from app.models.schemas import TokenData # Define TokenData schema
    # credentials_exception = HTTPException(
    #     status_code=status.HTTP_401_UNAUTHORIZED,
    #     detail="Could not validate credentials",
    #     headers={"WWW-Authenticate": "Bearer"},
    # )
    # try:
    #     payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    #     username: str = payload.get("sub")
    #     if username is None:
    #         raise credentials_exception
    #     token_data = TokenData(username=username)
    # except JWTError as e:
    #     logger.warning(f"JWT Error: {e}")
    #     raise credentials_exception
    # # --- Lookup user based on token_data.username ---
    # # user = get_user_from_db(username=token_data.username) # Implement this
    # # if user is None:
    # #    raise credentials_exception
    # # return user # Return the actual user object
    # --- End of JWT verification logic ---

    # Placeholder logic:
    if not token or not token.endswith("_valid_jwt_placeholder"): # Replace with real check
        logger.warning(f"Invalid placeholder token received: {token}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials (placeholder check failed)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Simulate returning a user identifier from the token
    user_id = token.split("_")[0]
    logger.debug(f"Placeholder token verified for user: {user_id}")
    return {"user_id": user_id, "roles": ["user"]} # Example user data


# --- Example Token Endpoint (Needs a separate router, e.g., routers/auth.py) ---
# from fastapi import APIRouter
# from fastapi.security import OAuth2PasswordRequestForm
# from datetime import timedelta
# # from app.security.auth import create_access_token # Need a token creation function

# auth_router = APIRouter(tags=["Authentication"])

# @auth_router.post(f"{settings.API_V1_STR}/auth/token", response_model=TokenSchema) # Define TokenSchema
# async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
#     # --- Replace with actual user authentication ---
#     # user = authenticate_user(form_data.username, form_data.password) # Implement this
#     # if not user:
#     #     raise HTTPException(
#     #         status_code=status.HTTP_401_UNAUTHORIZED,
#     #         detail="Incorrect username or password",
#     #         headers={"WWW-Authenticate": "Bearer"},
#     #     )
#     # --- End authentication ---
#
#     # Placeholder authentication:
#     if form_data.username == "testuser" and form_data.password == "password":
#         # Create JWT
#         # access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
#         # access_token = create_access_token(
#         #     data={"sub": user.username}, expires_delta=access_token_expires
#         # )
#         access_token = f"{form_data.username}_valid_jwt_placeholder" # Placeholder token
#         return {"access_token": access_token, "token_type": "bearer"}
#     else:
#          raise HTTPException(
#              status_code=status.HTTP_401_UNAUTHORIZED,
#              detail="Incorrect username or password",
#              headers={"WWW-Authenticate": "Bearer"},
#          )