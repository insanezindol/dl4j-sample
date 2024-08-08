import org.apache.commons.io.FilenameUtils;
import org.nd4j.common.resources.Downloader;

import java.io.File;
import java.net.URL;

public enum DownloaderUtility {

    DATAEXAMPLES("DataExamples.zip", "dl4j-examples", "e4de9c6f19aaae21fed45bfe2a730cbb", "2MB");

    private final String BASE_URL;
    private final String DATA_FOLDER;
    private final String ZIP_FILE;
    private final String MD5;
    private final String DATA_SIZE;
    private static final String AZURE_BLOB_URL = "https://dl4jdata.blob.core.windows.net/dl4j-examples";

    /*
     * 애저 블롭 스토리지에 업로드된 리소스와 함께 사용할 수 있습니다.
     */
    DownloaderUtility(String zipFile, String dataFolder, String md5, String dataSize) {
        this(AZURE_BLOB_URL + "/" + dataFolder, zipFile, dataFolder, md5, dataSize);
    }

    /*
     * 기본 URL에서 사용자의 홈 디렉토리 아래에 있는 지정된 디렉토리로 zip 파일을 다운로드합니다
     */
    DownloaderUtility(String baseURL, String zipFile, String dataFolder, String md5, String dataSize) {
        BASE_URL = baseURL;
        DATA_FOLDER = dataFolder;
        ZIP_FILE = zipFile;
        MD5 = md5;
        DATA_SIZE = dataSize;
    }

    public String Download() throws Exception {
        return Download(true);
    }

    public String Download(boolean returnSubFolder) throws Exception {
        String dataURL = BASE_URL + "/" + ZIP_FILE;
        String downloadPath = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), ZIP_FILE);
        String extractDir = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/" + DATA_FOLDER);
        if (!new File(extractDir).exists())
            new File(extractDir).mkdirs();
        String dataPathLocal = extractDir;
        if (returnSubFolder) {
            String resourceName = ZIP_FILE.substring(0, ZIP_FILE.lastIndexOf(".zip"));
            dataPathLocal = FilenameUtils.concat(extractDir, resourceName);
        }
        int downloadRetries = 10;
        if (!new File(dataPathLocal).exists() || new File(dataPathLocal).list().length == 0) {
            System.out.println("_______________________________________________________________________");
            System.out.println("Downloading data (" + DATA_SIZE + ") and extracting to \n\t" + dataPathLocal);
            System.out.println("_______________________________________________________________________");
            Downloader.downloadAndExtract("files",
                    new URL(dataURL),
                    new File(downloadPath),
                    new File(extractDir),
                    MD5,
                    downloadRetries);
        } else {
            System.out.println("_______________________________________________________________________");
            System.out.println("Example data present in \n\t" + dataPathLocal);
            System.out.println("_______________________________________________________________________");
        }
        return dataPathLocal;
    }
}
