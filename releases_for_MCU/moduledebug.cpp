#include <stdio.h>
#include "matrix/matrix_static.h"

int matrix_compare(const matrix_qfloat_data* a, const matrix_qfloat_data* b) {
	if (a->rows != b->rows || a->cols != b->cols) return 1;
	for (int i = 0; i < a->rows * a->cols; i++) {
		if (a->data[i] != b->data[i]) return 1;
	}
	return 0;
}

int main(){
	int error_count = 0;
	matrix_qfloat_data *mat1 = alloc_matrix_qfloat(),*mat2 = alloc_matrix_qfloat(),*excepted = alloc_matrix_qfloat(),*actual = alloc_matrix_qfloat();
	matrix_qfloat_init(mat1,2,10,0);
	matrix_qfloat_init(mat2,2,10,0);
	matrix_qfloat_init(excepted,2,10,0);
	matrix_qfloat_init(actual,2,10,0);

	qfloat madd1_0[] = { -163754,-272048,-865164,-1117611,-322304,652088,170396,-1208605,1192449,-741703,1120651,393795,158706,-835118,-412634,-1075990,27810,-335143,-464952,39792 };
	qfloat madd2_0[] = { 329228,-354261,-765402,243237,16115,-294701,41835,-1278835,466171,-22279,-280193,-1008786,-1154618,19724,1242434,1153963,745635,-999622,612297,935009 };
	qfloat mresu_0[] = { 165473,-626309,-1630566,-874374,-306189,357387,212231,-2487440,1658620,-763982,840458,-614991,-995911,-815395,829800,77973,773445,-1334765,147345,974801 };

	matrix_qfloat_reset(mat1,2,10,madd1_0);
	matrix_qfloat_reset(mat2,2,10,madd2_0);
	matrix_qfloat_reset(excepted,2,10,mresu_0);
	matrix_qfloat_add(mat1, mat2, actual);

	if(matrix_compare(excepted, actual) != 0) {
		printf("Add test 0 failed!\n");
		error_count++;
	}
	DbgPrint_qfloat_matrix(excepted,"excepted add 0");
	putc('\n',stdout);
	DbgPrint_qfloat_matrix(actual,"actual add 0");
	puts("\n");


	qfloat madd1_1[] = { 148539,892204,1290701,-1188854,-1223525,2918,-182094,1136992,-346926,1249540,1309417,550730,-249928,415136,-15827,264475,-1276033,-430468,-28488,1195575 };
	qfloat madd2_1[] = { -517040,-64006,-1297978,556207,-1105051,1289420,-125320,-165996,-914699,-891081,-1194736,-215020,-1098165,-861836,-18636,-878268,492383,-430577,268375,525816 };
	qfloat mresu_1[] = { -368501,828198,-7276,-632647,-2328576,1292338,-307414,970996,-1261625,358460,114681,335709,-1348092,-446701,-34463,-613793,-783650,-861045,239887,1721391 };

	matrix_qfloat_reset(mat1,2,10,madd1_1);
	matrix_qfloat_reset(mat2,2,10,madd2_1);
	matrix_qfloat_reset(excepted,2,10,mresu_1);
	matrix_qfloat_add(mat1, mat2, actual);

	if(matrix_compare(excepted, actual) != 0) {
		printf("Add test 1 failed!\n");
		error_count++;
	}
	DbgPrint_qfloat_matrix(excepted,"excepted add 1");
	putc('\n',stdout);
	DbgPrint_qfloat_matrix(actual,"actual add 1");
	puts("\n");


	qfloat madd1_2[] = { 907205,1270853,1008637,532453,333720,494941,1050329,508875,-919397,-325214,-584458,1274212,281904,-444610,-862465,-94374,461492,84446,1151938,-193055 };
	qfloat madd2_2[] = { 199557,-392942,1262971,-136894,870441,647927,1150264,46942,217010,128299,625104,839399,1207080,342720,985698,-607105,-1029417,1184871,365696,992892 };
	qfloat mresu_2[] = { 1106762,877911,2271609,395559,1204162,1142868,2200593,555817,-702387,-196915,40647,2113612,1488984,-101891,123233,-701480,-567925,1269317,1517634,799837 };

	matrix_qfloat_reset(mat1,2,10,madd1_2);
	matrix_qfloat_reset(mat2,2,10,madd2_2);
	matrix_qfloat_reset(excepted,2,10,mresu_2);
	matrix_qfloat_add(mat1, mat2, actual);

	if(matrix_compare(excepted, actual) != 0) {
		printf("Add test 2 failed!\n");
		error_count++;
	}
	DbgPrint_qfloat_matrix(excepted,"excepted add 2");
	putc('\n',stdout);
	DbgPrint_qfloat_matrix(actual,"actual add 2");
	puts("\n");


	qfloat madd1_3[] = { 73876,1238761,683024,-865922,119135,634941,-1256481,-661373,142012,849436,-360942,-156313,1021051,1020139,-612624,-128281,-1172230,567575,1246256,1193134 };
	qfloat madd2_3[] = { 844601,-803676,425313,-768562,454587,567467,557323,-72568,-972866,151178,-841088,1011682,-479808,-1185445,-162537,1175218,1137724,-1032779,-1231656,881386 };
	qfloat mresu_3[] = { 918477,435085,1108337,-1634484,573722,1202408,-699158,-733941,-830854,1000614,-1202029,855369,541243,-165307,-775162,1046937,-34506,-465204,14600,2074520 };

	matrix_qfloat_reset(mat1,2,10,madd1_3);
	matrix_qfloat_reset(mat2,2,10,madd2_3);
	matrix_qfloat_reset(excepted,2,10,mresu_3);
	matrix_qfloat_add(mat1, mat2, actual);

	if(matrix_compare(excepted, actual) != 0) {
		printf("Add test 3 failed!\n");
		error_count++;
	}
	DbgPrint_qfloat_matrix(excepted,"excepted add 3");
	putc('\n',stdout);
	DbgPrint_qfloat_matrix(actual,"actual add 3");
	puts("\n");


	qfloat madd1_4[] = { -201574,-1225448,361618,-10863,-995121,897027,601916,-11264,473145,-955720,-802719,1286060,-990716,990461,343294,1263100,351201,-1236955,637462,-803574 };
	qfloat madd2_4[] = { 855660,-753305,-771836,-672963,424414,361532,727049,52420,-295570,453839,-36994,-529471,-1046929,1009630,-962167,687438,892871,1133176,-706660,742541 };
	qfloat mresu_4[] = { 654086,-1978753,-410218,-683826,-570707,1258559,1328965,41156,177576,-501881,-839713,756588,-2037645,2000090,-618873,1950538,1244071,-103779,-69198,-61033 };

	matrix_qfloat_reset(mat1,2,10,madd1_4);
	matrix_qfloat_reset(mat2,2,10,madd2_4);
	matrix_qfloat_reset(excepted,2,10,mresu_4);
	matrix_qfloat_add(mat1, mat2, actual);

	if(matrix_compare(excepted, actual) != 0) {
		printf("Add test 4 failed!\n");
		error_count++;
	}
	DbgPrint_qfloat_matrix(excepted,"excepted add 4");
	putc('\n',stdout);
	DbgPrint_qfloat_matrix(actual,"actual add 4");
	puts("\n");


	qfloat madd1_0_2[] = { -485838,-1274227,-487430,959789,-955079,-4140 };
	qfloat madd2_0_2[] = { -42726,711482,-755356,1258470,-435509,-1018460,1309305,532612,789995,521941,-1003495,541239,276422,-975128,852377,745474,720493,-702918,-188520,-153053,-183115,969394,1199814,684446,-755402,611780,972227,-397410,-331173,-1034241 };
	qfloat mresu_0_2[] = { 21189790,-23007770,-8698569,4539537,-7725992,-11494398,-30945966,12674311,272090,6798792,14010111,2470901,-15166540,32598203,-18752395,-25818260,8613638,18069180,14337948,9939769 };

	matrix_qfloat_reset(mat1,2,3,madd1_0_2);
	matrix_qfloat_reset(mat2,3,10,madd2_0_2);
	matrix_qfloat_reset(excepted,2,10,mresu_0_2);
	matrix_qfloat_mulpty(mat1, mat2, actual);

	if(matrix_compare(excepted, actual) != 0) {
		printf("Multiply test 0 failed!\n");
		error_count++;
	}
	DbgPrint_qfloat_matrix(excepted,"excepted mulpty 0");
	putc('\n',stdout);
	DbgPrint_qfloat_matrix(actual,"actual mulpty 0");
	puts("\n");


	qfloat madd1_1_2[] = { -1195470,-842346,672498,1237496,613519,296272 };
	qfloat madd2_1_2[] = { 875398,1275353,-3425,538805,676717,37519,565159,69837,-1227159,-116493,-600929,58731,-891314,40439,1064574,-298254,115012,-238114,651630,-960255,562030,-247487,585749,170410,404506,-1156328,897580,-230322,590334,-1205777 };
	qfloat mresu_1_2[] = { -2477385,-26558714,17529365,-8599673,-21876591,-8716552,-2577067,-576852,20067319,2094252,13445037,23513078,-5760731,11323032,24572962,-7311133,15806154,-1951632,-14403041,-16640204 };

	matrix_qfloat_reset(mat1,2,3,madd1_1_2);
	matrix_qfloat_reset(mat2,3,10,madd2_1_2);
	matrix_qfloat_reset(excepted,2,10,mresu_1_2);
	matrix_qfloat_mulpty(mat1, mat2, actual);

	if(matrix_compare(excepted, actual) != 0) {
		printf("Multiply test 1 failed!\n");
		error_count++;
	}
	DbgPrint_qfloat_matrix(excepted,"excepted mulpty 1");
	putc('\n',stdout);
	DbgPrint_qfloat_matrix(actual,"actual mulpty 1");
	puts("\n");


	qfloat madd1_2_2[] = { -235926,576341,811612,1163817,1299762,-1271008 };
	qfloat madd2_2_2[] = { -472053,-884198,-1065835,-1116264,1228489,786457,1178723,55686,-941375,-567986,-272606,785800,940525,192190,1154490,-467233,951241,-943577,543652,49529,-339990,357661,530153,-407447,1288480,-954018,-774794,895059,1058649,-542737 };
	qfloat mresu_2_2[] = { -4908512,14522945,18673701,662759,21687197,-18754932,-5473085,2586069,21280454,-4241086,-7195666,-7053861,-10556135,-8109389,19723976,23201978,54824398,-35083693,-26466690,1421617 };

	matrix_qfloat_reset(mat1,2,3,madd1_2_2);
	matrix_qfloat_reset(mat2,3,10,madd2_2_2);
	matrix_qfloat_reset(excepted,2,10,mresu_2_2);
	matrix_qfloat_mulpty(mat1, mat2, actual);

	if(matrix_compare(excepted, actual) != 0) {
		printf("Multiply test 2 failed!\n");
		error_count++;
	}
	DbgPrint_qfloat_matrix(excepted,"excepted mulpty 2");
	putc('\n',stdout);
	DbgPrint_qfloat_matrix(actual,"actual mulpty 2");
	puts("\n");


	qfloat madd1_3_2[] = { 496499,-486275,1270344,906255,-821045,-1233330 };
	qfloat madd2_3_2[] = { -384704,-1205151,486251,338655,374260,-918433,1119903,-1059626,213228,-716895,673455,678026,988928,-1281182,907839,-621300,-41128,638975,-338825,1194612,-69599,-55625,561710,209471,302969,-210186,-784482,419571,858343,1191622 };
	qfloat mresu_3_2[] = { -9260608,-15239352,7234159,16132332,1971978,-6422223,-6416790,-4635958,20767511,8803131,-12447172,-24112869,-16236287,16791817,-11899760,-961178,30764946,-30554018,-8959817,-47305043 };

	matrix_qfloat_reset(mat1,2,3,madd1_3_2);
	matrix_qfloat_reset(mat2,3,10,madd2_3_2);
	matrix_qfloat_reset(excepted,2,10,mresu_3_2);
	matrix_qfloat_mulpty(mat1, mat2, actual);

	if(matrix_compare(excepted, actual) != 0) {
		printf("Multiply test 3 failed!\n");
		error_count++;
	}
	DbgPrint_qfloat_matrix(excepted,"excepted mulpty 3");
	putc('\n',stdout);
	DbgPrint_qfloat_matrix(actual,"actual mulpty 3");
	puts("\n");


	qfloat madd1_4_2[] = { 761624,789389,210136,-582173,1138682,-582120 };
	qfloat madd2_4_2[] = { -729151,314476,463336,-744339,954923,-618173,-1277436,-366881,-477988,640840,1143763,-1056554,493860,-988500,143109,528893,-267906,-915198,1061317,-1101621,-383535,-234776,-552130,706998,849382,56464,-137700,254770,490643,1236922 };
	qfloat mresu_4_2[] = { 4073174,-9824431,9562887,-18289966,15544845,-632454,-18514165,-14470461,8801975,-1855558,29756734,-19065728,9369105,-16842799,-13540897,14179318,7916067,-14905364,18328272,-35820202 };

	matrix_qfloat_reset(mat1,2,3,madd1_4_2);
	matrix_qfloat_reset(mat2,3,10,madd2_4_2);
	matrix_qfloat_reset(excepted,2,10,mresu_4_2);
	matrix_qfloat_mulpty(mat1, mat2, actual);

	if(matrix_compare(excepted, actual) != 0) {
		printf("Multiply test 4 failed!\n");
		error_count++;
	}
	DbgPrint_qfloat_matrix(excepted,"excepted mulpty 4");
	putc('\n',stdout);
	DbgPrint_qfloat_matrix(actual,"actual mulpty 4");
	puts("\n");


	printf("Total errors: %d\n", error_count);
	return error_count;
}