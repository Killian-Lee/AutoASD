document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const defaultUploadText = document.getElementById('defaultUploadText');
    const fileInfo = document.getElementById('fileInfo');
    const deleteFile = document.getElementById('deleteFile');
    
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    const closeBtn = document.getElementsByClassName('modal-close')[0];

    // 为所有图片添加点击事件
    document.querySelectorAll('.graph-img').forEach(img => {
        img.onclick = function() {
            modal.style.display = "flex";
            modalImg.src = this.src;
        }
    });

    // 点击关闭按钮关闭模态框
    closeBtn.onclick = function() {
        modal.style.display = "none";
    }

    // 点击模态框外部关闭
    modal.onclick = function(event) {
        if (event.target === modal) {
            modal.style.display = "none";
        }
    }

    // ESC键关闭模态框
    document.addEventListener('keydown', function(event) {
        if (event.key === "Escape") {
            modal.style.display = "none";
        }
    });

    // Drag and drop upload
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    // 显示文件信息
    function showFileInfo(file) {
        const fileName = fileInfo.querySelector('.file-name');
        const fileSize = fileInfo.querySelector('.file-size');
        
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        
        defaultUploadText.style.display = 'none';
        fileInfo.style.display = 'block';
        uploadBtn.disabled = false;
    }

    // 清除文件
    function clearFile() {
        fileInput.value = '';
        defaultUploadText.style.display = 'block';
        fileInfo.style.display = 'none';
        uploadBtn.disabled = true;
    }

    // 格式化文件大小
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // 文件选择处理
    fileInput.addEventListener('change', function(e) {
        if (this.files[0]) {
            showFileInfo(this.files[0]);
        }
    });

    // 删除文件
    deleteFile.addEventListener('click', function(e) {
        e.stopPropagation();
        clearFile();
    });

    // 拖拽上传
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        fileInput.files = e.dataTransfer.files;
        if (fileInput.files[0]) {
            showFileInfo(fileInput.files[0]);
        }
    });

    // 初始化禁用上传按钮
    uploadBtn.disabled = true;

    // File upload and analysis handling
    uploadBtn.addEventListener('click', async () => {
        if (!fileInput.files[0]) {
            alert('Please select a file first');
            return;
        }

        // 显示所有加载动画
        document.querySelectorAll('.loading-animation').forEach(loading => {
            loading.style.display = 'flex';
        });
        
        // 禁用上传按钮
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Analyzing...';

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.success) {
                updateVisualizations(data);
            }
        } catch (error) {
            console.error('Upload failed:', error);
        } finally {
            // 隐藏所有加载动画
            document.querySelectorAll('.loading-animation').forEach(loading => {
                loading.style.display = 'none';
            });
            
            // 恢复上传按钮
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload and Analysis';
        }
    });

    // Update all visualizations
    function updateVisualizations(data) {
        const baseOption = {
            tooltip: {},
            series: [{
                type: 'graph',
                layout: 'force',
                roam: true,
                label: {
                    show: true
                },
                force: {
                    repulsion: 100
                }
            }]
        };

        // Update each visualization with its specific data
        groundTruthChart.setOption({
            ...baseOption,
            title: { text: 'Ground Truth' },
            series: [{ ...baseOption.series[0], data: data.groundTruth.nodes, links: data.groundTruth.links }]
        });

        ratioChart.setOption({
            ...baseOption,
            title: { text: 'RATIO' },
            series: [{ ...baseOption.series[0], data: data.ratio.nodes, links: data.ratio.links }]
        });

        antibenfordChart.setOption({
            ...baseOption,
            title: { text: 'AntiBenford' },
            series: [{ ...baseOption.series[0], data: data.antibenford.nodes, links: data.antibenford.links }]
        });

        clareChart.setOption({
            ...baseOption,
            title: { text: 'CLARE' },
            series: [{ ...baseOption.series[0], data: data.clare.nodes, links: data.clare.links }]
        });
    }

    // Handle window resize
    window.addEventListener('resize', () => {
        groundTruthChart.resize();
        ratioChart.resize();
        antibenfordChart.resize();
        clareChart.resize();
    });
}); 