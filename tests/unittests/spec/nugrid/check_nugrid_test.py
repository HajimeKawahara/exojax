from exojax.spec.check_nugrid import check_scale_xsmode


def test_check_scale_xsmode():
    assert check_scale_xsmode("lpf") == "ESLOG"
    assert check_scale_xsmode("modit") == "ESLOG"
    assert check_scale_xsmode("premodit") == "ESLOG"
    assert check_scale_xsmode("presolar") == "ESLOG"
    assert check_scale_xsmode("dit") == "ESLIN"
    assert check_scale_xsmode("LPF") == "ESLOG"
    assert check_scale_xsmode("MODIT") == "ESLOG"
    assert check_scale_xsmode("PREMODIT") == "ESLOG"
    assert check_scale_xsmode("PRESOLAR") == "ESLOG"
    assert check_scale_xsmode("DIT") == "ESLIN"


if __name__ == "__main__":
    test_check_scale_xsmode()