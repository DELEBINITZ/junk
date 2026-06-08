"""Test RBAC enforcement."""

from security_intel.security.rbac import SecurityContext, check_rbac


def test_admin_has_all_roles():
    sc = SecurityContext(org_id="org1", user_id="u1", roles=("admin",))
    assert sc.has_role("viewer")
    assert sc.has_role("analyst")
    assert sc.has_role("admin")


def test_viewer_limited():
    sc = SecurityContext(org_id="org1", user_id="u1", roles=("viewer",))
    assert sc.has_role("viewer")
    assert not sc.has_role("analyst")
    assert not sc.has_role("admin")


def test_check_rbac_function():
    assert check_rbac(["analyst"], "viewer") is True
    assert check_rbac(["viewer"], "analyst") is False
    assert check_rbac(["admin"], "admin") is True
