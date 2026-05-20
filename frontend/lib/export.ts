import { toPng } from 'html-to-image';
import jsPDF from 'jspdf';
import { toast } from 'sonner';

export const exportDashboardAsImage = async (elementId: string, filename: string = 'hydra-terminal-capture') => {
  const element = document.getElementById(elementId);
  if (!element) {
    toast.error('Dashboard element not found for export');
    return;
  }

  const id = toast.loading('Generating image...');

  try {
    const dataUrl = await toPng(element, {
      backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--background').trim(),
      style: {
        borderRadius: '0',
      },
      cacheBust: true,
    });

    const link = document.createElement('a');
    link.download = `${filename}.png`;
    link.href = dataUrl;
    link.click();
    toast.success('Dashboard exported as PNG', { id });
  } catch (error) {
    console.error('Export failed', error);
    toast.error('Failed to export as image', { id });
  }
};

export const exportDashboardAsPdf = async (elementId: string, filename: string = 'hydra-terminal-report') => {
  const element = document.getElementById(elementId);
  if (!element) {
    toast.error('Dashboard element not found for export');
    return;
  }

  const id = toast.loading('Generating PDF...');

  try {
    const dataUrl = await toPng(element, {
      backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--background').trim(),
      cacheBust: true,
    });

    const pdf = new jsPDF({
      orientation: 'landscape',
      unit: 'px',
      format: [element.offsetWidth, element.offsetHeight],
    });

    pdf.addImage(dataUrl, 'PNG', 0, 0, element.offsetWidth, element.offsetHeight);
    pdf.save(`${filename}.pdf`);
    toast.success('Dashboard exported as PDF', { id });
  } catch (error) {
    console.error('Export failed', error);
    toast.error('Failed to export as PDF', { id });
  }
};
