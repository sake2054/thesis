export async function apiGet(path, options = {}) {
  return apiRequest(path, { method: "GET", ...options });
}

export async function apiPost(path, body, options = {}) {
  return apiRequest(path, {
    method: "POST",
    body: JSON.stringify(body),
    ...options
  });
}

export async function apiPatch(path, body, options = {}) {
  return apiRequest(path, {
    method: "PATCH",
    body: JSON.stringify(body),
    ...options
  });
}

export async function apiRequest(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {})
  };
  const response = await fetch(path, {
    ...options,
    headers
  });
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch {
      // Keep the generic HTTP message.
    }
    throw new Error(message);
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

export async function downloadAdminCsv(table, pin) {
  const response = await fetch(`/api/admin/export/${table}.csv`, {
    headers: {
      "x-admin-pin": pin
    }
  });
  if (!response.ok) {
    throw new Error(`CSV export failed: ${response.status}`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${table}.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
