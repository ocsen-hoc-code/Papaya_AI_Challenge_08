from medextract.postprocessing.roles import check_person_roles, reconcile_date_labels
from medextract.schemas import RawField


def _dates(adm_value, adm_evidence, dis_value, dis_evidence):
    return {
        "admission_date": RawField(value=adm_value, confidence=0.9, evidence=adm_evidence),
        "discharge_date": RawField(value=dis_value, confidence=0.9, evidence=dis_evidence),
    }


def test_swaps_dates_when_evidence_carries_the_other_label_thai():
    fields = _dates(
        "12/04/2567",
        "วันที่ถึงหน่วย 12/04/2567",  # misread label, no role keyword
        "15/04/2567",
        "วันที่รับไว้ในโรงพยาบาล 15/04/2567",  # admission label on the discharge field
    )
    issues = reconcile_date_labels(fields)
    assert fields["admission_date"].value == "15/04/2567"
    assert fields["discharge_date"].value == "12/04/2567"
    assert [i.code for i in issues] == ["fields_swapped_by_label"]
    assert issues[0].severity == "warning"


def test_swaps_dates_english_and_vietnamese_labels():
    en = _dates("2024-03-18", "Date of Discharge: 18/03/2024", "2024-03-11", "Admitted: 11/03/2024")
    assert reconcile_date_labels(en) and en["admission_date"].value == "2024-03-11"
    vi = _dates("18/03/2024", "Ra viện lúc: 18/03/2024", "11/03/2024", "Vào viện lúc: 11/03/2024")
    assert reconcile_date_labels(vi) and vi["discharge_date"].value == "18/03/2024"


def test_keeps_dates_when_labels_match_or_are_missing():
    ok = _dates("11/03/2024", "Vào viện: 11/03/2024", "18/03/2024", "Ra viện: 18/03/2024")
    assert reconcile_date_labels(ok) == [] and ok["admission_date"].value == "11/03/2024"
    unlabeled = _dates("11/03/2024", "11/03/2024", "18/03/2024", None)
    assert reconcile_date_labels(unlabeled) == []
    # Printed order is wrong but labels are right: keep it, validation reports date_order.
    wrong_order = _dates("15/04/2024", "Admission 15/04/2024", "12/04/2024", "Discharge 12/04/2024")
    assert reconcile_date_labels(wrong_order) == [] and wrong_order["admission_date"].value == "15/04/2024"


def test_one_crossed_label_with_own_label_on_other_field_is_left_alone():
    fields = _dates("11/03/2024", "Admission: 11/03/2024", "11/03/2024", "Admission: 11/03/2024")
    assert reconcile_date_labels(fields) == []


def test_missing_date_fields_are_ignored():
    assert reconcile_date_labels({}) == []
    fields = _dates(None, None, "12/04/2024", "Admission 12/04/2024")
    assert reconcile_date_labels(fields) == []


def test_person_labelled_as_director_in_text_layer_is_rejected():
    text = "Bác sĩ điều trị\n(Ký, ghi rõ họ tên)\n\nGIÁM ĐỐC\n(Ký, ghi rõ họ tên, đóng dấu)\nBS.CKII Trần Văn A\n"
    fields = {
        "attending_physician": RawField(value="BS.CKII Trần Văn A", confidence=0.97, evidence="BS.CKII Trần Văn A")
    }
    issues = check_person_roles(fields, text)
    assert fields["attending_physician"].value is None
    assert [i.code for i in issues] == ["role_mismatch"]
    assert "Trần Văn A" in issues[0].message


def test_person_role_from_evidence_without_text_layer():
    fields = {"doctor_name": RawField(value="ภก.สมศักดิ์ ดีมาก", evidence="เภสัชกรผู้จ่ายยา ภก.สมศักดิ์ ดีมาก")}
    assert check_person_roles(fields, None)
    assert fields["doctor_name"].value is None


def test_real_attending_physician_is_kept():
    text = "Attending physician\nDr. Sarah Lim\nMedical Records\n"
    fields = {
        "attending_physician": RawField(
            value="Dr. Sarah Lim", confidence=0.95, evidence="Attending physician Dr. Sarah Lim"
        )
    }
    assert check_person_roles(fields, text) == []
    assert fields["attending_physician"].value == "Dr. Sarah Lim"
    assert check_person_roles({"doctor_name": RawField(value=None)}, text) == []
