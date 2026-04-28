$(document).ready(function() {
    // 1. ตั้งค่า Select2
    $('.select2').select2({
        theme: 'bootstrap-5',
        width: '100%'
    });

    // 2. เมื่อเลือกหมอ -> Auto-select แผนก
    $('#doctorSelect').on('change', function() {
        const spec = $(this).find(':selected').data('spec');
        if (spec) {
            $('#specialtySelect').val(spec).trigger('change');
        }
    });

    // 3. เมื่อแผนกเปลี่ยน -> กรองหัตถการ
    $('#specialtySelect').on('change', function() {
        const selectedSpec = $(this).val();
        const $txSelect = $('#treatmentSelect');
        
        $txSelect.empty();
        
        if (selectedSpec) {
            $txSelect.prop('disabled', false);
            const filtered = treatmentsData.filter(t => t.spec === selectedSpec);
            
            filtered.forEach(t => {
                const option = new Option(t.text, t.id, false, false);
                $txSelect.append(option);
            });
        } else {
            $txSelect.prop('disabled', true);
            $txSelect.append('<option value="">-- กรุณาเลือกแผนกก่อน --</option>');
        }
        $txSelect.trigger('change');
    });

    // ตั้งค่าเวลาปัจจุบันเริ่มต้น
    const now = new Date();
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    const startTimeInput = document.getElementById('startTimeInput');
    if (startTimeInput) {
        startTimeInput.value = now.toISOString().slice(0, 16);
    }
});