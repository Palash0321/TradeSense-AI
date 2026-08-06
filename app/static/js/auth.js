function getToken() {

    return localStorage.getItem("access_token");

}

function isLoggedIn() {

    return getToken() !== null;

}

function logout() {

    localStorage.removeItem("access_token");

    localStorage.removeItem("user");

    window.location = "/login";

}

async function getCurrentUser() {

    const token = getToken();

    if (!token) {

        return null;

    }

    const response = await fetch("/auth/me", {

        headers: {

            "Authorization": "Bearer " + token

        }

    });

    if (!response.ok) {

        logout();

        return null;

    }

    return await response.json();

}